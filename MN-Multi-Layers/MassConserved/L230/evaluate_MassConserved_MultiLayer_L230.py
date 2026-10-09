import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

ARCH = "L230"
H1, H2, H3 = 2, 3, 0

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
from MCPBRNN_lib_tools.MCNZoo_MultiLayers import MCN_unit
from MCPBRNN_lib_tools.headLayers import MC_Linear_Layer

METRICS = ["NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss"]
SPLITS = ("train", "selection", "testing", "spinup")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate MassConserved multi-layer MCP architecture L230."
    )
    p.add_argument("--c_mean", type=float, default=412.9139085)
    p.add_argument("--c_std", type=float, default=77.49943867)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )
    return p.parse_args()


def resolve_path(value, default):
    path = Path(value) if value else default
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


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

    f = torch.tensor(flags.values, device=device)

    masks = {
        "train": f.eq(-1).unsqueeze(1),
        "selection": f.eq(0).unsqueeze(1),
        "testing": f.eq(1).unsqueeze(1),
        "spinup": f.eq(-99999).unsqueeze(1),
    }

    return x, y, masks


def calc_metrics(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()

    kge, corr, a, b, kgess = KGE(sim, obs)

    def safe(value):
        return -99999.0 if np.isnan(value) else float(value)

    return {
        "NSE": float(NS(sim, obs)),
        "KGE": safe(kge),
        "KGE-A": float(a),
        "KGE-B": float(b),
        "Corr": safe(corr),
        "mse": float(mean_squared_error(obs, sim)),
        "KGEss": safe(kgess),
    }


def build_time_axis(n):
    spin = pd.date_range("1948-10-01", "1949-09-30", freq="D")
    sim = pd.date_range("1948-10-01", "1988-09-30", freq="D")

    time, phase = [], []

    for i in range(1, 4):
        time += list(spin.strftime("%Y-%m-%d"))
        phase += [f"spinup{i}"] * len(spin)

    time += list(sim.strftime("%Y-%m-%d"))
    phase += ["simulation"] * len(sim)

    if len(time) != n:
        raise ValueError(f"Expected {len(time)} rows but found {n}.")

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


def evaluate(model, x, y, masks, c_mean, c_std):
    model.eval()

    with torch.no_grad():
        result = model(x, c_mean, c_std)

    row = {}

    for split in SPLITS:
        sim = torch.masked_select(result[0], masks[split]).unsqueeze(1)
        obs = torch.masked_select(y, masks[split]).unsqueeze(1)

        suffix = "" if split == "train" else f"_{split}"

        for name, value in calc_metrics(sim, obs).items():
            row[f"{name}{suffix}"] = value

    return result, row


def print_skill_metrics(row):
    print("\\nSkill metrics")
    for split in SPLITS:
        suffix = "" if split == "train" else f"_{split}"
        print(
            f"{split:9s} | "
            + " | ".join(
                f"{metric}={row[f'{metric}{suffix}']:.4f}"
                for metric in METRICS
            )
        )


class Model(nn.Module):
    """MassConserved L230: 2-3-0."""

    def __init__(self):
        super().__init__()

        self.MCPBRNNNode_Layer1 = MCN_unit(
            input_size=1, hidden_size=H1, initial_forget_bias=0
        )

        self.MCPBRNNNode_Layer12_linear = MC_Linear_Layer(
            input_size=1, hidden_size_old=H1, hidden_size_new=H2
        )

        self.MCPBRNNNode_Layer2 = MCN_unit(
            input_size=1, hidden_size=H2, initial_forget_bias=0
        )

        self.MCPBRNNNode_Layer2out_linear = MC_Linear_Layer(
            input_size=1, hidden_size_old=H2, hidden_size_new=1
        )

    def forward(self, x, c_mean, c_std):
        h1, c1, l1, lc1, bp1, gi1, go1, gl1, glc1, gr1 = (
            self.MCPBRNNNode_Layer1(x, c_mean, c_std)
        )

        mapped12 = self.MCPBRNNNode_Layer12_linear(h1)
        x2 = torch.cat((mapped12, x[:, :, 1]), dim=1).unsqueeze(1)

        h2, c2, l2, lc2, bp2, gi2, go2, gl2, glc2, gr2 = (
            self.MCPBRNNNode_Layer2(x2, c_mean, c_std)
        )

        total_q = self.MCPBRNNNode_Layer2out_linear(h2)

        return (
            total_q,
            h1, h2,
            c1, c2,
            l1, l2,
            lc1, lc2,
            bp1, bp2,
            gi1, gi2,
            go1, go2,
            gl1, gl2,
            glc1, glc2,
            gr1, gr2,
            mapped12, total_q,
        )


def save_outputs(out_dir, result, n):
    time, phase = build_time_axis(n)
    layer_columns = ["layer1_node1", "layer1_node2", "layer2_node1", "layer2_node2", "layer2_node3"]

    combined = {
        "node_discharge.csv": torch.cat((result[1], result[2]), dim=1),
        "storage.csv": torch.cat((result[3], result[4]), dim=1),
        "loss_unconstrained.csv": torch.cat((result[5], result[6]), dim=1),
        "loss_constrained.csv": torch.cat((result[7], result[8]), dim=1),
        "input_bypass.csv": torch.cat((result[9], result[10]), dim=1),
        "input_gate.csv": torch.cat((result[11], result[12]), dim=1),
        "output_gate.csv": torch.cat((result[13], result[14]), dim=1),
        "loss_gate.csv": torch.cat((result[15], result[16]), dim=1),
        "loss_gate_constrained.csv": torch.cat((result[17], result[18]), dim=1),
        "remember_gate.csv": torch.cat((result[19], result[20]), dim=1),
    }

    save_series(out_dir / "discharge.csv", result[0], time, phase, ["discharge"])

    for filename, tensor in combined.items():
        save_series(out_dir / filename, tensor, time, phase, layer_columns)

    save_series(
        out_dir / "interlayer_mapping.csv",
        torch.cat((result[21], result[22]), dim=1),
        time,
        phase,
        ["layer1_to_layer2_node1", "layer1_to_layer2_node2", "layer1_to_layer2_node3", "layer2_to_output"],
    )


def main():
    args = parse_args()

    device = resolve_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint)

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
        model, x, y, masks, args.c_mean, args.c_std
    )

    pd.DataFrame([row]).to_csv(
        out_dir / "metrics.csv",
        index=False,
    )

    print_skill_metrics(row)
    save_outputs(out_dir, result, x.shape[0])

    print(f"\nArchitecture: {ARCH} = {H1}-{H2}-{H3}")
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
