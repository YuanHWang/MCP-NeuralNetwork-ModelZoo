import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start):
    for path in (start, *start.parents):
        if (path / "MCPBRNN_lib_tools").is_dir():
            return path
    raise RuntimeError(
        f"Could not find project root containing MCPBRNN_lib_tools above {start}."
    )


PROJECT_DIR = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS


HIDDEN_SIZE = 5
SPIN_LEN = 1095
TRAIN_TIME_LEN = 8400
METRICS = ("NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss")
SPLITS = ("train", "selection", "testing", "spinup")

# Exact scenario numbering used by the historical scripts:
# 1 removed node -> 5 combinations
# 2 removed nodes -> 10 combinations
# 3 removed nodes -> 10 combinations
# 4 removed nodes -> 5 combinations
PRUNE_NODE_COMBINATIONS = {
    1: {
        1: (1,),
        2: (2,),
        3: (3,),
        4: (4,),
        5: (5,),
    },
    2: {
        1: (5, 1),
        2: (5, 2),
        3: (5, 3),
        4: (5, 4),
        5: (1, 2),
        6: (1, 3),
        7: (1, 4),
        8: (2, 3),
        9: (2, 4),
        10: (3, 4),
    },
    3: {
        1: (3, 4, 5),
        2: (2, 4, 5),
        3: (2, 3, 5),
        4: (2, 3, 4),
        5: (1, 4, 5),
        6: (1, 3, 5),
        7: (1, 3, 4),
        8: (1, 2, 5),
        9: (1, 2, 4),
        10: (1, 2, 3),
    },
    4: {
        1: (2, 3, 4, 5),
        2: (1, 3, 4, 5),
        3: (1, 2, 4, 5),
        4: (1, 2, 3, 5),
        5: (1, 2, 3, 4),
    },
}


def parse_common_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Default: the only .pt file beside this script.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help=(
            "Folder containing LeafRiverDaily_43YR.txt and "
            "LeafRiverDaily_43YR_Flag.txt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output folder. A case-specific default is used when omitted.",
    )
    parser.add_argument(
        "--prune",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Number of output flow paths to remove.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=0,
        help=(
            "Combination index for the selected pruning level. "
            "Valid counts are 5/10/10/5 for prune=1/2/3/4."
        ),
    )
    parser.add_argument(
        "--c_mean",
        type=float,
        default=412.9139085,
    )
    parser.add_argument(
        "--c_std",
        type=float,
        default=77.49943867,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )
    return parser


def resolve_path(value, default):
    path = Path(value) if value else default
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def resolve_checkpoint(value):
    if value:
        path = resolve_path(value, SCRIPT_DIR)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    checkpoints = sorted(SCRIPT_DIR.glob("*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one .pt checkpoint in {SCRIPT_DIR}, "
            f"but found {len(checkpoints)}."
        )
    return checkpoints[0]


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def get_pruned_nodes(prune, scenario):
    if prune == 0:
        if scenario not in (0, 1):
            raise ValueError(
                "For --prune 0 use --scenario 0 "
                "(scenario 1 is also accepted for legacy convenience)."
            )
        return ()

    valid = PRUNE_NODE_COMBINATIONS[prune]
    if scenario not in valid:
        allowed = ", ".join(str(v) for v in sorted(valid))
        raise ValueError(
            f"Invalid scenario {scenario} for prune={prune}. "
            f"Valid scenarios: {allowed}."
        )
    return valid[scenario]


def base_output_weights(model):
    return torch.softmax(
        model.MCPBRNNNode.wo[:, 0],
        dim=0,
    )


def effective_output_weights(model, pruned_nodes):
    """Remove selected output paths and renormalize the remaining DS weights."""
    weights = base_output_weights(model)

    if not pruned_nodes:
        return weights

    mask = torch.ones_like(weights)
    index = torch.tensor(
        [node - 1 for node in pruned_nodes],
        dtype=torch.long,
        device=weights.device,
    )
    mask[index] = 0.0

    weights = weights * mask
    denominator = weights.sum()

    if denominator <= 0:
        raise RuntimeError("Pruning removed every output path.")

    return weights / denominator


def aggregate_node_discharge(node_discharge, weights):
    return torch.mm(
        node_discharge,
        weights.unsqueeze(1),
    )


def disconnect_shared_cell_states(model, pruned_nodes):
    """Fully disconnect pruned SALO cell states from retained-node gates.

    In the SALO class, rows of weight_b1_yom / weight_b1_ylm correspond
    to source cell-state information. Zeroing row j prevents state j from
    entering every node's output/loss gate. Combined with removal of its
    output path, this makes the pruned node functionally irrelevant to
    the retained network.
    """
    if not pruned_nodes:
        return

    module = model.MCPBRNNNode

    with torch.no_grad():
        for node in pruned_nodes:
            j = node - 1
            module.weight_b1_yom[j, :] = 0.0
            module.weight_b1_ylm[j, :] = 0.0


def load_data(data_dir, device):
    data = pd.read_csv(
        data_dir / "LeafRiverDaily_43YR.txt",
        sep=r"\s+",
        header=None,
        names=["P", "PET", "Q"],
    )
    flags = pd.read_csv(
        data_dir / "LeafRiverDaily_43YR_Flag.txt",
        sep=r"\s+",
        header=None,
        names=["Flag"],
    )["Flag"]

    if len(data) != len(flags):
        raise ValueError("Data and skill-flag files have different lengths.")

    x = torch.tensor(
        data[["P", "PET"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    y = torch.tensor(
        data[["Q"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    flag_t = torch.tensor(
        flags.to_numpy(),
        device=device,
    )

    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }

    return data, flags, x, y, masks


def safe_value(value):
    return -99999.0 if np.isnan(value) else float(value)


def calculate_metrics(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()

    kge, corr, kge_a, kge_b, kgess = KGE(sim, obs)

    return {
        "NSE": safe_value(NS(sim, obs)),
        "KGE": safe_value(kge),
        "KGE-A": safe_value(kge_a),
        "KGE-B": safe_value(kge_b),
        "Corr": safe_value(corr),
        "mse": safe_value(mean_squared_error(obs, sim)),
        "KGEss": safe_value(kgess),
    }


def skill_table(prediction, y, masks):
    rows = []

    for split in SPLITS:
        sim = torch.masked_select(
            prediction,
            masks[split],
        ).unsqueeze(1)
        obs = torch.masked_select(
            y,
            masks[split],
        ).unsqueeze(1)

        row = {
            "split": split,
            "n": int(sim.numel()),
        }
        row.update(calculate_metrics(sim, obs))
        rows.append(row)

    return pd.DataFrame(rows)


def print_metrics(metrics):
    print("\nSkill metrics")
    for _, row in metrics.iterrows():
        print(
            f"{row['split']:9s} | "
            + " | ".join(
                f"{metric}={row[metric]:.4f}"
                for metric in METRICS
            )
        )


def build_time_axis(n_rows):
    spinup_dates = pd.date_range(
        "1948-10-01",
        "1949-09-30",
        freq="D",
    )
    simulation_dates = pd.date_range(
        "1948-10-01",
        "1988-09-30",
        freq="D",
    )

    time = []
    phase = []

    for cycle in range(1, 4):
        time.extend(spinup_dates.strftime("%Y-%m-%d"))
        phase.extend([f"spinup{cycle}"] * len(spinup_dates))

    time.extend(simulation_dates.strftime("%Y-%m-%d"))
    phase.extend(["simulation"] * len(simulation_dates))

    if len(time) != n_rows:
        raise ValueError(
            f"Paper-3 timeline expects {len(time)} rows, found {n_rows}."
        )

    return time, phase


def save_matrix(path, tensor, time, phase, prefix):
    values = tensor.detach().cpu().numpy()

    if values.ndim == 1:
        values = values[:, None]

    frame = pd.DataFrame(
        values,
        columns=[
            f"{prefix}_{i + 1}"
            for i in range(values.shape[1])
        ],
    )
    frame.insert(0, "phase", phase)
    frame.insert(0, "time", time)
    frame.to_csv(path, index=False)


def save_outputs(
    output_dir,
    model,
    raw_result,
    prediction,
    base_weights,
    effective_weights,
    pruned_nodes,
    data,
    flags,
):
    time, phase = build_time_axis(len(data))

    save_matrix(
        output_dir / "discharge.csv",
        prediction,
        time,
        phase,
        "discharge",
    )
    save_matrix(
        output_dir / "node_discharge.csv",
        raw_result[1],
        time,
        phase,
        "node_discharge",
    )
    save_matrix(
        output_dir / "storage.csv",
        raw_result[2],
        time,
        phase,
        "storage",
    )
    save_matrix(
        output_dir / "loss_unconstrained.csv",
        raw_result[3],
        time,
        phase,
        "loss_unconstrained",
    )
    save_matrix(
        output_dir / "loss_constrained.csv",
        raw_result[4],
        time,
        phase,
        "loss_constrained",
    )
    save_matrix(
        output_dir / "input_bypass.csv",
        raw_result[5],
        time,
        phase,
        "input_bypass",
    )
    save_matrix(
        output_dir / "input_gate.csv",
        raw_result[6],
        time,
        phase,
        "input_gate",
    )
    save_matrix(
        output_dir / "output_gate.csv",
        raw_result[7],
        time,
        phase,
        "output_gate",
    )
    save_matrix(
        output_dir / "loss_gate.csv",
        raw_result[8],
        time,
        phase,
        "loss_gate",
    )
    save_matrix(
        output_dir / "loss_gate_constrained.csv",
        raw_result[9],
        time,
        phase,
        "loss_gate_constrained",
    )
    save_matrix(
        output_dir / "remember_gate.csv",
        raw_result[10],
        time,
        phase,
        "remember_gate",
    )

    weighted = (
        raw_result[1]
        * effective_weights.unsqueeze(0)
    )
    save_matrix(
        output_dir / "weighted_node_discharge.csv",
        weighted,
        time,
        phase,
        "weighted_node_discharge",
    )

    pruned_set = set(pruned_nodes)
    pd.DataFrame(
        {
            "node": np.arange(1, HIDDEN_SIZE + 1),
            "pruned": [
                node in pruned_set
                for node in range(1, HIDDEN_SIZE + 1)
            ],
            "base_output_weight": (
                base_weights.detach().cpu().numpy()
            ),
            "effective_output_weight": (
                effective_weights.detach().cpu().numpy()
            ),
        }
    ).to_csv(
        output_dir / "output_weights.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "time": time,
            "phase": phase,
            "P": data["P"].to_numpy(),
            "PET": data["PET"].to_numpy(),
            "Q": data["Q"].to_numpy(),
            "skill_flag": flags.to_numpy(),
        }
    ).to_csv(
        output_dir / "forcing_observation.csv",
        index=False,
    )

from MCPBRNN_lib_tools.prune_class import (
    MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_shareoutputlossGate_LossGateUpdate,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.MCPBRNNNode = (
            MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_shareoutputlossGate_LossGateUpdate(
                input_size=1,
                hidden_size=HIDDEN_SIZE,
                gate_dim=1,
                gate_dim_ucorr=1,
                spinLen=SPIN_LEN,
                traintimeLen=TRAIN_TIME_LEN,
                initial_forget_bias=0,
            )
        )

    def forward(self, x, y, c_mean, c_std):
        return self.MCPBRNNNode(
            x,
            0,
            0,
            y,
            c_mean,
            c_std,
        )


def main():
    parser = parse_common_args(
        "Evaluate Figure 13e: full SALO pruning (flow path + shared state)."
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint)
    pruned_nodes = get_pruned_nodes(args.prune, args.scenario)

    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )
    output_dir = resolve_path(
        args.output_dir,
        SCRIPT_DIR
        / "evaluation_full"
        / f"prune_{args.prune}_scenario_{args.scenario}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data, flags, x, y, masks = load_data(data_dir, device)

    model = Model().to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)

    # PAPER DEFINITION:
    # Full removal = selected output flow paths are removed AND the
    # corresponding cell states no longer provide shared information
    # to the gates of retained nodes.
    disconnect_shared_cell_states(
        model,
        pruned_nodes,
    )

    model.eval()
    with torch.no_grad():
        raw_result = model(
            x,
            y,
            args.c_mean,
            args.c_std,
        )
        base_weights = base_output_weights(model)
        weights = effective_output_weights(
            model,
            pruned_nodes,
        )
        prediction = aggregate_node_discharge(
            raw_result[1],
            weights,
        )

    metrics = skill_table(prediction, y, masks)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    print_metrics(metrics)

    save_outputs(
        output_dir,
        model,
        raw_result,
        prediction,
        base_weights,
        weights,
        pruned_nodes,
        data,
        flags,
    )

    retained = tuple(
        node
        for node in range(1, HIDDEN_SIZE + 1)
        if node not in set(pruned_nodes)
    )

    pd.DataFrame(
        {
            "prune": [args.prune],
            "scenario": [args.scenario],
            "pruned_output_paths": [
                ",".join(map(str, pruned_nodes))
            ],
            "disconnected_shared_cell_states": [
                ",".join(map(str, pruned_nodes))
            ],
            "retained_nodes": [
                ",".join(map(str, retained))
            ],
            "interpretation": [
                "full SALO pruning: output path and its shared state influence removed"
            ],
        }
    ).to_csv(
        output_dir / "pruning_configuration.csv",
        index=False,
    )

    print(
        f"\nFully pruned nodes/paths: "
        f"{pruned_nodes if pruned_nodes else 'none'}"
    )
    print(f"Retained nodes: {retained}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Outputs:    {output_dir}")


if __name__ == "__main__":
    main()
