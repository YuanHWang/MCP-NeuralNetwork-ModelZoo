import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

ARCH = "L400-SALO"
HIDDEN_SIZE = 4
MODEL_LABEL = "shared loss/output parallel-node"

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
from MCPBRNN_lib_tools.miscellaneous import MCN_SingleMultiML_SharedLossOutput_IC


METRICS = ["NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss"]
SPLITS = ("train", "selection", "testing", "spinup")


def parse_args():
    p = argparse.ArgumentParser(
        description=f"Evaluate {MODEL_LABEL} MCP architecture {ARCH}."
    )
    p.add_argument("--case_no", type=int, default=0)
    p.add_argument("--c_mean", type=float, default=412.9139085)
    p.add_argument("--c_std", type=float, default=77.49943867)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return p.parse_args()


def resolve_checkpoint(value, case_no):
    if value:
        path = Path(value)

        if not path.is_absolute():
            path = (SCRIPT_DIR / path).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        return path

    direct = sorted(SCRIPT_DIR.glob("best_model_epoch*.pt"))

    if len(direct) == 1:
        return direct[0]

    case_dir = SCRIPT_DIR / f"case_{case_no}"
    nested = sorted(case_dir.glob("best_model_epoch*.pt"))

    if len(nested) == 1:
        return nested[0]

    fallback = sorted(SCRIPT_DIR.glob("*.pt"))

    if len(fallback) == 1:
        return fallback[0]

    raise RuntimeError(
        "Could not determine checkpoint automatically. "
        f"Found {len(direct)} best-model checkpoints in {SCRIPT_DIR} "
        f"and {len(nested)} in {case_dir}. "
        "Use --checkpoint to specify the file explicitly."
    )


def resolve_path(value, default):
    path = Path(value) if value else default
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return torch.device(name)


def load_data(data_dir, device):
    data_file = data_dir / "LeafRiverDaily_43YR.txt"
    flag_file = data_dir / "LeafRiverDaily_43YR_Flag.txt"

    if not data_file.is_file():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if not flag_file.is_file():
        raise FileNotFoundError(f"Flag file not found: {flag_file}")

    data = pd.read_csv(
        data_file,
        sep=r"\s+",
        header=None,
        names=["P", "PET", "Q"],
    )

    flags = pd.read_csv(
        flag_file,
        sep=r"\s+",
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

    flag = torch.tensor(flags.values, device=device)

    masks = {
        "train": flag.eq(-1).unsqueeze(1),
        "selection": flag.eq(0).unsqueeze(1),
        "testing": flag.eq(1).unsqueeze(1),
        "spinup": flag.eq(-99999).unsqueeze(1),
    }

    return x, y, masks


def calc_metrics(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()

    kge, corr, alpha, beta, kgess = KGE(sim, obs)

    def safe(value):
        return -99999.0 if np.isnan(value) else float(value)

    return {
        "NSE": float(NS(sim, obs)),
        "KGE": safe(kge),
        "KGE-A": float(alpha),
        "KGE-B": float(beta),
        "Corr": safe(corr),
        "mse": float(mean_squared_error(obs, sim)),
        "KGEss": safe(kgess),
    }


def evaluate(model, x, y, masks, c_mean, c_std):
    model.eval()

    with torch.no_grad():
        result = model(x, c_mean, c_std)

    # Historical scripts clipped negative total discharge for skill metrics.
    q_metric = result[0].clamp_min(0.0)

    row = {}

    for split in SPLITS:
        sim = torch.masked_select(q_metric, masks[split]).unsqueeze(1)
        obs = torch.masked_select(y, masks[split]).unsqueeze(1)

        suffix = "" if split == "train" else f"_{split}"

        for name, value in calc_metrics(sim, obs).items():
            row[f"{name}{suffix}"] = value

    return result, row


def print_skill_metrics(row):
    print("\nSkill metrics")

    for split in SPLITS:
        suffix = "" if split == "train" else f"_{split}"

        print(
            f"{split:9s} | "
            + " | ".join(
                f"{metric}={row[f'{metric}{suffix}']:.4f}"
                for metric in METRICS
            )
        )


def build_time_axis(n):
    spin = pd.date_range("1948-10-01", "1949-09-30", freq="D")
    simulation = pd.date_range("1948-10-01", "1988-09-30", freq="D")

    time, phase = [], []

    for i in range(1, 4):
        time += list(spin.strftime("%Y-%m-%d"))
        phase += [f"spinup{i}"] * len(spin)

    time += list(simulation.strftime("%Y-%m-%d"))
    phase += ["simulation"] * len(simulation)

    if len(time) != n:
        raise ValueError(
            f"Expected {len(time)} rows from the Leaf River time axis, "
            f"but model output has {n} rows."
        )

    return time, phase


def save_series(path, tensor, time, phase, columns):
    array = tensor.detach().cpu().numpy()

    if array.ndim == 1:
        array = array[:, None]

    if array.shape[1] != len(columns):
        raise ValueError(
            f"{path.name}: tensor has {array.shape[1]} columns, "
            f"but {len(columns)} names were provided."
        )

    df = pd.DataFrame(array, columns=columns)
    df.insert(0, "time", time)
    df["phase"] = phase
    df.to_csv(path, index=False)


def save_outputs(out_dir, result, n):
    time, phase = build_time_axis(n)
    node_columns = [
        f"node{i}" for i in range(1, HIDDEN_SIZE + 1)
    ]

    outputs = {
        "node_discharge.csv": result[1],
        "storage.csv": result[2],
        "loss_unconstrained.csv": result[3],
        "loss_constrained.csv": result[4],
        "input_bypass.csv": result[5],
        "input_gate.csv": result[6],
        "output_gate.csv": result[7],
        "loss_gate.csv": result[8],
        "loss_gate_constrained.csv": result[9],
        "remember_gate.csv": result[10],
    }

    save_series(
        out_dir / "discharge.csv",
        result[0],
        time,
        phase,
        ["discharge"],
    )

    for filename, tensor in outputs.items():
        save_series(
            out_dir / filename,
            tensor,
            time,
            phase,
            node_columns,
        )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.MCPBRNNNode = MCN_SingleMultiML_SharedLossOutput_IC(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            initial_forget_bias=0,
        )

    def forward(self, x, c_mean, c_std):
        return self.MCPBRNNNode(x, c_mean, c_std)


def main():
    args = parse_args()

    device = resolve_device(args.device)
    checkpoint = resolve_checkpoint(
        args.checkpoint,
        args.case_no,
    )

    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )

    out_dir = resolve_path(
        args.output_dir,
        SCRIPT_DIR / "evaluation",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, masks = load_data(data_dir, device)

    model = Model().to(device)

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)

    result, row = evaluate(
        model,
        x,
        y,
        masks,
        args.c_mean,
        args.c_std,
    )

    pd.DataFrame([row]).to_csv(
        out_dir / "metrics.csv",
        index=False,
    )

    print_skill_metrics(row)
    save_outputs(out_dir, result, x.shape[0])

    print(f"\nArchitecture: {ARCH}")
    print(f"Model: {MODEL_LABEL}")
    print(f"Nodes: {HIDDEN_SIZE}")
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
