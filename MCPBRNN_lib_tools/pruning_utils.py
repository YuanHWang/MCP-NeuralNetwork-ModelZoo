import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "MCPBRNN_lib_tools").is_dir():
            return path
    raise RuntimeError(
        f"Could not find project root containing MCPBRNN_lib_tools above {start}."
    )


PROJECT_DIR = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS  # noqa: E402
from MCPBRNN_lib_tools.prune_class import (  # noqa: E402
    MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_LossGateUpdate,
    MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_shareoutputlossGate_LossGateUpdate,
)


HIDDEN_SIZE = 5
SPIN_LEN = 1095
TRAIN_TIME_LEN = 8400
METRICS = ("NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss")
SPLITS = ("train", "selection", "testing", "spinup")

CASE_INFO = {
    "case1": {
        "title": "Case 1: non-sharing DS pruning",
        "checkpoint_dir": "Prune_Case1",
        "output_dir": "case1_nonsharing",
        "sharing": False,
        "full": False,
    },
    "partial": {
        "title": "Case 2-P: SALO partial pruning",
        "checkpoint_dir": "Prune_Case2",
        "output_dir": "case2_partial",
        "sharing": True,
        "full": False,
    },
    "full": {
        "title": "Case 2-F: SALO full pruning",
        "checkpoint_dir": "Prune_Case2",
        "output_dir": "case2_full",
        "sharing": True,
        "full": True,
    },
}


class Model(nn.Module):
    def __init__(self, sharing: bool):
        super().__init__()
        cls = (
            MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_shareoutputlossGate_LossGateUpdate
            if sharing
            else MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_LossGateUpdate
        )
        self.MCPBRNNNode = cls(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            gate_dim=1,
            gate_dim_ucorr=1,
            spinLen=SPIN_LEN,
            traintimeLen=TRAIN_TIME_LEN,
            initial_forget_bias=0,
        )

    def forward(self, x, y, c_mean, c_std):
        return self.MCPBRNNNode(x, 0, 0, y, c_mean, c_std)


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path. If omitted, exactly one .pt file must exist "
            "in the corresponding Prune_Case1 or Prune_Case2 folder."
        ),
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
        "--prune_count",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=0,
        help=(
            "Automatically prune this many flow paths, beginning with the "
            "smallest learned output weight."
        ),
    )
    parser.add_argument(
        "--prune_nodes",
        default=None,
        help=(
            "Optional manual override using 1-based node IDs, e.g. '2,4'. "
            "When provided, --prune_count is ignored."
        ),
    )
    parser.add_argument("--c_mean", type=float, default=412.9139085)
    parser.add_argument("--c_std", type=float, default=77.49943867)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def resolve_path(value, default):
    path = Path(value) if value else Path(default)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def resolve_checkpoint(value, checkpoint_dir):
    if value:
        path = resolve_path(value, SCRIPT_DIR)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    case_dir = SCRIPT_DIR / checkpoint_dir
    checkpoints = sorted(case_dir.glob("*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one .pt checkpoint in {case_dir}, "
            f"but found {len(checkpoints)}. Use --checkpoint to choose one."
        )
    return checkpoints[0]


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


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
    flag_t = torch.tensor(flags.to_numpy(), device=device)

    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }
    return data, flags, x, y, masks


def load_state_dict(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def learned_output_weights(model):
    """Return the learned DS flow-path weights A, B, C, ... ."""
    return torch.softmax(model.MCPBRNNNode.wo[:, 0], dim=0)


def parse_manual_nodes(value):
    if value is None:
        return None

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--prune_nodes was provided but no node IDs were found.")

    nodes = tuple(int(token) for token in tokens)
    if len(set(nodes)) != len(nodes):
        raise ValueError("--prune_nodes contains duplicate node IDs.")
    if any(node < 1 or node > HIDDEN_SIZE for node in nodes):
        raise ValueError(f"Node IDs must be between 1 and {HIDDEN_SIZE}.")
    if len(nodes) >= HIDDEN_SIZE:
        raise ValueError("At least one output flow path must remain active.")
    return nodes


def select_pruned_nodes(base_weights, prune_count, prune_nodes):
    manual = parse_manual_nodes(prune_nodes)
    if manual is not None:
        return manual, "manual"

    if prune_count == 0:
        return (), "none"

    order = torch.argsort(base_weights, descending=False)
    nodes = tuple((order[:prune_count] + 1).detach().cpu().tolist())
    return nodes, "smallest learned output weights"


def mask_output_weights(base_weights, pruned_nodes):
    """Mask selected learned flow-path weights WITHOUT renormalization."""
    weights = base_weights.clone()
    if pruned_nodes:
        index = torch.tensor(
            [node - 1 for node in pruned_nodes],
            dtype=torch.long,
            device=weights.device,
        )
        weights[index] = 0.0
    return weights


def disconnect_shared_interactions(model, pruned_nodes):
    """Remove selected source-state interactions from SALO output/loss gates.

    In the shared classes, gate inputs are computed as storage @ W. Therefore
    row j of W contains the influence of source state j on all target gates.
    Zeroing that row removes the selected state from the shared interaction
    terms without changing the learned output weights or renormalizing them.
    """
    if not pruned_nodes:
        return

    module = model.MCPBRNNNode
    with torch.no_grad():
        for node in pruned_nodes:
            j = node - 1
            module.weight_b1_yom[j, :] = 0.0
            module.weight_b1_ylm[j, :] = 0.0


def aggregate_node_discharge(node_discharge, output_weights):
    return torch.mm(node_discharge, output_weights.unsqueeze(1))


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
        sim = torch.masked_select(prediction, masks[split]).unsqueeze(1)
        obs = torch.masked_select(y, masks[split]).unsqueeze(1)
        row = {"split": split, "n": int(sim.numel())}
        row.update(calculate_metrics(sim, obs))
        rows.append(row)
    return pd.DataFrame(rows)


def print_metrics(metrics):
    print("\nSkill metrics")
    for _, row in metrics.iterrows():
        print(
            f"{row['split']:9s} | "
            + " | ".join(f"{metric}={row[metric]:.4f}" for metric in METRICS)
        )


def print_weights(base_weights, pruned_nodes):
    values = base_weights.detach().cpu().numpy()
    order = np.argsort(values) + 1
    print("\nLearned flow-path output weights")
    for node, value in enumerate(values, start=1):
        mark = "  <-- pruned" if node in set(pruned_nodes) else ""
        print(f"node {node}: {value:.8f}{mark}")
    print("smallest -> largest:", " -> ".join(map(str, order.tolist())))


def build_time_axis(n_rows):
    spinup_dates = pd.date_range("1948-10-01", "1949-09-30", freq="D")
    simulation_dates = pd.date_range("1948-10-01", "1988-09-30", freq="D")

    time = []
    phase = []
    for cycle in range(1, 4):
        time.extend(spinup_dates.strftime("%Y-%m-%d"))
        phase.extend([f"spinup{cycle}"] * len(spinup_dates))
    time.extend(simulation_dates.strftime("%Y-%m-%d"))
    phase.extend(["simulation"] * len(simulation_dates))

    if len(time) != n_rows:
        raise ValueError(
            f"Expected the Paper-3 Leaf River timeline to have {len(time)} rows, "
            f"but found {n_rows}."
        )
    return time, phase


def save_matrix(path, tensor, time, phase, prefix):
    values = tensor.detach().cpu().numpy()
    if values.ndim == 1:
        values = values[:, None]
    frame = pd.DataFrame(
        values,
        columns=[f"{prefix}_{i + 1}" for i in range(values.shape[1])],
    )
    frame.insert(0, "phase", phase)
    frame.insert(0, "time", time)
    frame.to_csv(path, index=False)


def save_outputs(
    output_dir,
    case,
    model,
    result,
    prediction,
    base_weights,
    masked_weights,
    pruned_nodes,
    selection_rule,
    data,
    flags,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    time, phase = build_time_axis(len(data))

    save_matrix(output_dir / "discharge.csv", prediction, time, phase, "discharge")
    save_matrix(output_dir / "node_discharge.csv", result[1], time, phase, "node_discharge")
    save_matrix(output_dir / "storage.csv", result[2], time, phase, "storage")
    save_matrix(output_dir / "loss_unconstrained.csv", result[3], time, phase, "loss_unconstrained")
    save_matrix(output_dir / "loss_constrained.csv", result[4], time, phase, "loss_constrained")
    save_matrix(output_dir / "input_bypass.csv", result[5], time, phase, "input_bypass")
    save_matrix(output_dir / "input_gate.csv", result[6], time, phase, "input_gate")
    save_matrix(output_dir / "output_gate.csv", result[7], time, phase, "output_gate")
    save_matrix(output_dir / "loss_gate.csv", result[8], time, phase, "loss_gate")
    save_matrix(output_dir / "loss_gate_constrained.csv", result[9], time, phase, "loss_gate_constrained")
    save_matrix(output_dir / "remember_gate.csv", result[10], time, phase, "remember_gate")

    weighted = result[1] * masked_weights.unsqueeze(0)
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
            "pruned": [node in pruned_set for node in range(1, HIDDEN_SIZE + 1)],
            "base_output_weight": base_weights.detach().cpu().numpy(),
            "masked_output_weight": masked_weights.detach().cpu().numpy(),
        }
    ).to_csv(output_dir / "output_weights.csv", index=False)

    pd.DataFrame(
        {
            "case": [case],
            "selection_rule": [selection_rule],
            "pruned_nodes": [",".join(map(str, pruned_nodes))],
            "retained_weight_sum": [float(masked_weights.sum().detach().cpu())],
            "renormalized_after_pruning": [False],
            "shared_interactions_removed": [case == "full"],
        }
    ).to_csv(output_dir / "pruning_configuration.csv", index=False)

    pd.DataFrame(
        {
            "time": time,
            "phase": phase,
            "P": data["P"].to_numpy(),
            "PET": data["PET"].to_numpy(),
            "Q": data["Q"].to_numpy(),
            "skill_flag": flags.to_numpy(),
        }
    ).to_csv(output_dir / "forcing_observation.csv", index=False)

    if CASE_INFO[case]["sharing"]:
        module = model.MCPBRNNNode
        pd.DataFrame(module.weight_b1_yom.detach().cpu().numpy()).to_csv(
            output_dir / "output_gate_state_interactions.csv",
            index_label="source_state_index_0based",
        )
        pd.DataFrame(module.weight_b1_ylm.detach().cpu().numpy()).to_csv(
            output_dir / "loss_gate_state_interactions.csv",
            index_label="source_state_index_0based",
        )


def default_output_dir(case, pruned_nodes, selection_rule):
    info = CASE_INFO[case]
    if not pruned_nodes:
        suffix = "baseline"
    elif selection_rule == "manual":
        suffix = "manual_nodes_" + "-".join(map(str, pruned_nodes))
    else:
        suffix = f"smallest_{len(pruned_nodes)}"
    return SCRIPT_DIR / "evaluation" / info["output_dir"] / suffix


def run_case(case):
    info = CASE_INFO[case]
    args = parse_args(info["title"])
    device = resolve_device(args.device)

    checkpoint = resolve_checkpoint(args.checkpoint, info["checkpoint_dir"])
    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )

    data, flags, x, y, masks = load_data(data_dir, device)

    model = Model(sharing=info["sharing"]).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device), strict=True)

    # A, B, C, ... are the learned DS output weights used to rank pathways.
    base_weights = learned_output_weights(model)
    pruned_nodes, selection_rule = select_pruned_nodes(
        base_weights,
        args.prune_count,
        args.prune_nodes,
    )

    # Full pruning additionally removes the selected source-state interactions
    # from both shared output and loss gates. Partial pruning leaves these
    # interaction matrices untouched.
    if info["full"]:
        disconnect_shared_interactions(model, pruned_nodes)

    # Output paths are masked AFTER the original Softmax weights are obtained.
    # The retained A, B, C, ... values are never renormalized.
    masked_weights = mask_output_weights(base_weights, pruned_nodes)

    model.eval()
    with torch.no_grad():
        result = model(x, y, args.c_mean, args.c_std)
        prediction = aggregate_node_discharge(result[1], masked_weights)

    metrics = skill_table(prediction, y, masks)

    output_dir = resolve_path(
        args.output_dir,
        default_output_dir(case, pruned_nodes, selection_rule),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    print(f"\n{info['title']}")
    print_weights(base_weights, pruned_nodes)
    print(f"selection rule: {selection_rule}")
    print(f"retained weight sum: {masked_weights.sum().item():.8f}")
    print("output-weight renormalization: NO")
    if case == "partial":
        print("SALO shared interactions: RETAINED")
    elif case == "full":
        print("SALO shared interactions from pruned states: REMOVED")
    print_metrics(metrics)

    save_outputs(
        output_dir,
        case,
        model,
        result,
        prediction,
        base_weights,
        masked_weights,
        pruned_nodes,
        selection_rule,
        data,
        flags,
    )

    print(f"\nCheckpoint: {checkpoint}")
    print(f"Pruned nodes/paths: {pruned_nodes if pruned_nodes else 'none'}")
    print(f"Outputs: {output_dir}")
