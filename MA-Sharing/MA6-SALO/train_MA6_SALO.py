import argparse
import copy
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
        f"Could not find project root containing MCPBRNN_lib_tools above {start}"
    )


PROJECT_DIR = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS
from MCPBRNN_lib_tools.HyMOD_Shared_Cases import MA6_PETconstraint_CellShared
from MCPBRNN_lib_tools.Loss_Function import KGELoss


METRICS = ["NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss"]
SPLITS = ("train", "selection", "testing", "spinup")


def resolve_path(value, default):
    path = Path(value) if value else default
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def resolve_checkpoint(value):
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = (SCRIPT_DIR / path).resolve()
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


def load_state(path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    return state


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


def calculate_lag_kge(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()
    n = obs.shape[0]

    result = {}
    for lag in (1, 2, 3):
        kge, *_ = KGE(sim[lag:n], obs[: n - lag])
        result[f"KGEtimelag_{lag}"] = safe_value(kge)

    return result


def load_data(data_dir, device):
    data = pd.read_csv(
        data_dir / "LeafRiverDaily_43YR.txt",
        sep=r"\\s+",
        header=None,
        names=["P", "PET", "Q"],
    )
    flags = pd.read_csv(
        data_dir / "LeafRiverDaily_43YR_Flag.txt",
        sep=r"\\s+",
        header=None,
        names=["Flag"],
    )["Flag"]

    if len(data) != len(flags):
        raise ValueError("Data and flag files have different lengths.")

    x = torch.tensor(
        data[["P", "PET"]].values,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    y = torch.tensor(
        data[["Q"]].values,
        dtype=torch.float32,
        device=device,
    )

    flag_t = torch.tensor(flags.values, device=device)

    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }

    return data, flags, x, y, masks


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
        raise ValueError(f"Expected {len(time)} rows, but found {n_rows}.")

    return time, phase


def save_series(path, tensor, time, phase, column_name, clamp_nonnegative=False):
    values = tensor.detach().cpu()

    if clamp_nonnegative:
        values = torch.clamp(values, min=0.0)

    pd.DataFrame(
        {
            "time": time,
            column_name: values.numpy().reshape(-1),
            "phase": phase,
        }
    ).to_csv(path, index=False)


def print_metrics(row):
    print("\\nSkill metrics")

    for split in SPLITS:
        suffix = "" if split == "train" else f"_{split}"

        values = " | ".join(
            f"{metric}={row[f'{metric}{suffix}']:.4f}"
            for metric in METRICS
        )

        print(f"{split:9s} | {values}")

PARAMETER_COLUMNS = [
    "MCPBRNNNode.weight_r_yom",
    "MCPBRNNNode.weight_r_yogw",
    "MCPBRNNNode.weight_r_yofp",
    "MCPBRNNNode.weight_r_ylm",
    "MCPBRNNNode.weight_r_yfm",
    "MCPBRNNNode.bias_b0_yom",
    "MCPBRNNNode.weight_b1_yom",
    "MCPBRNNNode.weight_b2_yom",
    "MCPBRNNNode.weight_b3_yom",
    "MCPBRNNNode.bias_b0_yogw",
    "MCPBRNNNode.weight_b1_yogw",
    "MCPBRNNNode.weight_b2_yogw",
    "MCPBRNNNode.weight_b3_yogw",
    "MCPBRNNNode.bias_b0_yofp",
    "MCPBRNNNode.weight_b1_yofp",
    "MCPBRNNNode.weight_b2_yofp",
    "MCPBRNNNode.weight_b3_yofp",
    "MCPBRNNNode.bias_b0_ylm",
    "MCPBRNNNode.weight_b2_ylm",
    "MCPBRNNNode.weight_r_yom_sw",
    "MCPBRNNNode.weight_r_yfm_sw",
    "MCPBRNNNode.bias_b0_yom_sw",
    "MCPBRNNNode.weight_b1_yom_sw",
    "MCPBRNNNode.weight_b2_yom_sw",
    "MCPBRNNNode.weight_b3_yom_sw",
    "MCPBRNNNode.weight_r_yom_gw",
    "MCPBRNNNode.weight_r_yfm_gw",
    "MCPBRNNNode.bias_b0_yom_gw",
    "MCPBRNNNode.weight_b1_yom_gw",
    "MCPBRNNNode.weight_b2_yom_gw",
    "MCPBRNNNode.weight_b3_yom_gw",
]

class Model(nn.Module):
    """MA6-SALO wrapper retaining the original checkpoint module name."""

    def __init__(self):
        super().__init__()

        self.MCPBRNNNode = MA6_PETconstraint_CellShared(
            input_size=1,
            hidden_size=1,
            initial_forget_bias=0,
        )

    def forward(self, x, c_mean, c_std, sw_mean, sw_std, gw_mean, gw_std, ini_c):
        node = self.MCPBRNNNode(
            x,
            c_mean,
            c_std,
            sw_mean,
            sw_std,
            gw_mean,
            gw_std,
            ini_c,
        )

        return (
            node[2] + node[14] + node[18],  # system discharge
            node[0],                        # main discharge
            node[14],                       # SW discharge
            node[18],                       # GW discharge
            node[3],                        # main storage
            node[15],                       # SW storage
            node[19],                       # GW storage
            node[4],                        # loss unconstrained
            node[5],                        # loss constrained
            node[1],                        # recharge flux
            node[6],                        # input bypass
            node[7],                        # input gate
            node[8],                        # main output gate
            node[16],                       # SW output gate
            node[20],                       # GW output gate
            node[11],                       # loss gate
            node[12],                       # constrained loss gate
            node[13],                       # main remember gate
            node[17],                       # SW remember gate
            node[21],                       # GW remember gate
            node[9],                        # recharge gate
            node[2],                        # fast-path discharge
            node[10],                       # fast-path gate
        )

def evaluate_model(model, x, y, masks, c_mean, c_std, sw_mean, sw_std, gw_mean, gw_std, ini_c):
    model.eval()

    with torch.no_grad():
        result = model(x, c_mean, c_std, sw_mean, sw_std, gw_mean, gw_std, ini_c)

    row = {}

    for split in SPLITS:
        sim = torch.masked_select(result[0], masks[split]).unsqueeze(1)
        obs = torch.masked_select(y, masks[split]).unsqueeze(1)

        suffix = "" if split == "train" else f"_{split}"

        for name, value in calculate_metrics(sim, obs).items():
            row[f"{name}{suffix}"] = value

    train_sim = torch.masked_select(result[0], masks["train"]).unsqueeze(1)
    train_obs = torch.masked_select(y, masks["train"]).unsqueeze(1)
    row.update(calculate_lag_kge(train_sim, train_obs))

    return result, row

def save_outputs(output_dir, result, n_rows):
    time, phase = build_time_axis(n_rows)

    outputs = {
        "discharge.csv": (result[0], "discharge", True),
        "main_discharge.csv": (result[1], "main_discharge", False),
        "sw_discharge.csv": (result[2], "sw_discharge", False),
        "gw_discharge.csv": (result[3], "gw_discharge", False),
        "main_storage.csv": (result[4], "main_storage", False),
        "sw_storage.csv": (result[5], "sw_storage", False),
        "gw_storage.csv": (result[6], "gw_storage", False),
        "loss_unconstrained.csv": (result[7], "loss_unconstrained", False),
        "loss_constrained.csv": (result[8], "loss_constrained", False),
        "recharge.csv": (result[9], "recharge", False),
        "input_bypass.csv": (result[10], "input_bypass", False),
        "input_gate.csv": (result[11], "input_gate", False),
        "main_output_gate.csv": (result[12], "main_output_gate", False),
        "sw_output_gate.csv": (result[13], "sw_output_gate", False),
        "gw_output_gate.csv": (result[14], "gw_output_gate", False),
        "loss_gate.csv": (result[15], "loss_gate", False),
        "loss_gate_constrained.csv": (result[16], "loss_gate_constrained", False),
        "main_remember_gate.csv": (result[17], "main_remember_gate", False),
        "sw_remember_gate.csv": (result[18], "sw_remember_gate", False),
        "gw_remember_gate.csv": (result[19], "gw_remember_gate", False),
        "recharge_gate.csv": (result[20], "recharge_gate", False),
        "fast_path.csv": (result[21], "fast_path", False),
        "fast_path_gate.csv": (result[22], "fast_path_gate", False),
    }

    for filename, (tensor, column_name, clamp_nonnegative) in outputs.items():
        save_series(
            output_dir / filename,
            tensor,
            time,
            phase,
            column_name,
            clamp_nonnegative=clamp_nonnegative,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Continue training Paper-3 MA6-SALO from an existing checkpoint."
    )
    p.add_argument("--case_no", type=int, default=0)
    p.add_argument("--epoch_no", type=int, default=3000)
    p.add_argument("--seed_no", type=int, default=2925, help="Base seed; actual seed = seed_no + case_no.")
    p.add_argument("--c_mean", type=float, default=412.9139085)
    p.add_argument("--c_std", type=float, default=77.49943867)
    p.add_argument("--SW_mean", type=float, default=3.051248249)
    p.add_argument("--SW_std", type=float, default=3.365261828)
    p.add_argument("--GW_mean", type=float, default=1.87176748)
    p.add_argument("--GW_std", type=float, default=2.085571043)
    p.add_argument("--Ini_C", type=float, default=1.6871063)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if args.epoch_no < 1:
        raise ValueError("epoch_no must be at least 1.")

    if args.c_std == 0 or args.SW_std == 0 or args.GW_std == 0:
        raise ValueError("All storage normalization standard deviations must be non-zero.")

    seed = args.seed_no + args.case_no
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint)

    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )
    output_dir = resolve_path(
        args.output_dir,
        SCRIPT_DIR / f"case_{args.case_no}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, x, y, masks = load_data(data_dir, device)

    model = Model().to(device)
    state = load_state(checkpoint, device)

    model.load_state_dict(state, strict=True)
    print(f"Loaded training checkpoint: {checkpoint}")

    # Continue from model weights only; optimizer state starts fresh.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    loss_func = KGELoss()

    learning_rates = {300: 0.0125, 600: 0.0125}

    history = []
    best_epoch = None
    best_kge = -np.inf
    best_state = None

    for epoch in range(1, args.epoch_no + 1):
        if epoch in learning_rates:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch]

        model.train()
        optimizer.zero_grad()

        predictions = model(
            x,
            args.c_mean,
            args.c_std,
            args.SW_mean,
            args.SW_std,
            args.GW_mean,
            args.GW_std,
            args.Ini_C,
        )[0]

        sim = torch.masked_select(predictions, masks["train"]).unsqueeze(1)
        obs = torch.masked_select(y, masks["train"]).unsqueeze(1)

        loss = loss_func(sim, obs)
        loss.backward()
        optimizer.step()

        result, row = evaluate_model(
            model,
            x,
            y,
            masks,
            args.c_mean,
            args.c_std,
            args.SW_mean,
            args.SW_std,
            args.GW_mean,
            args.GW_std,
            args.Ini_C,
        )

        row["epoch"] = epoch
        current_state = model.state_dict()

        for name in PARAMETER_COLUMNS:
            row[name] = (
                current_state[name]
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )

        history.append(row)

        if row["KGE_selection"] > best_kge:
            best_kge = row["KGE_selection"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:4d} | "
            f"KGE train={row['KGE']:.4f} | "
            f"selection={row['KGE_selection']:.4f} | "
            f"testing={row['KGE_testing']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("No best model was selected.")

    history_columns = (
        ["epoch"]
        + PARAMETER_COLUMNS
        + METRICS
        + [
            f"{metric}_{split}"
            for split in ("selection", "testing", "spinup")
            for metric in METRICS
        ]
        + ["KGEtimelag_1", "KGEtimelag_2", "KGEtimelag_3"]
    )

    pd.DataFrame(history)[history_columns].to_csv(
        output_dir / "metrics_by_epoch.csv",
        index=False,
    )

    best_file = output_dir / f"best_model_epoch{best_epoch}.pt"
    torch.save(best_state, best_file)

    model.load_state_dict(best_state, strict=True)

    result, row = evaluate_model(
        model,
            x,
            y,
            masks,
            args.c_mean,
            args.c_std,
            args.SW_mean,
            args.SW_std,
            args.GW_mean,
            args.GW_std,
            args.Ini_C,
    )

    pd.DataFrame([row]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    print_metrics(row)
    save_outputs(output_dir, result, x.shape[0])

    pd.DataFrame([{"Best_Epoch": best_epoch}]).to_csv(
        output_dir / "best_epoch.csv",
        index=False,
    )

    print(f"\ncase_no={args.case_no} | seed={seed}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best KGE_selection: {best_kge:.6f}")
    print(f"Saved best model: {best_file}")


if __name__ == "__main__":
    main()
