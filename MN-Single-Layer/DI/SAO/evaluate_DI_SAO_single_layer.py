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
    for p in (start, *start.parents):
        if (p / "MCPBRNN_lib_tools").is_dir():
            return p
    raise RuntimeError("Could not find project root containing MCPBRNN_lib_tools.")

PROJECT_DIR = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS
from MCPBRNN_lib_tools.MCP_Net import MCPNet_DI_SAO_SingleLayer

METRICS = ["NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss"]
SPLITS = ("train", "selection", "testing", "spinup")

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate single-layer DI-SAO MCP network.")
    p.add_argument("--hidden_size", type=int, choices=[2, 3, 4, 5], default=2)
    p.add_argument("--c_mean", type=float, default=412.9139085)
    p.add_argument("--c_std", type=float, default=77.49943867)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()

def resolve_path(value, default):
    p = Path(value) if value else default
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()

def resolve_checkpoint(value, hidden_size):
    if value:
        return resolve_path(value, SCRIPT_DIR)
    folder = SCRIPT_DIR / f"{hidden_size}nodes"
    files = sorted(folder.glob("*.pt"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one .pt checkpoint in {folder}, but found {len(files)}.")
    return files[0]

def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)

def load_data(data_dir, device):
    data = pd.read_csv(data_dir / "LeafRiverDaily_43YR.txt", sep=r"\s+", header=None, names=["P", "PET", "Q"])
    flags = pd.read_csv(data_dir / "LeafRiverDaily_43YR_Flag.txt", sep=r"\s+", header=None, names=["Flag"])["Flag"]
    if len(data) != len(flags):
        raise ValueError("Data and flag files have different lengths.")

    x = torch.tensor(data[["P", "PET"]].values, dtype=torch.float32, device=device).unsqueeze(1)
    y = torch.tensor(data[["Q"]].values, dtype=torch.float32, device=device)
    f = torch.tensor(flags.values, device=device)
    masks = {
        "train": f.eq(-1).unsqueeze(1), "selection": f.eq(0).unsqueeze(1),
        "testing": f.eq(1).unsqueeze(1), "spinup": f.eq(-99999).unsqueeze(1),
    }
    return x, y, masks

def calc_metrics(sim_t, obs_t):
    sim, obs = sim_t.detach().cpu().numpy(), obs_t.detach().cpu().numpy()
    kge, corr, a, b, kgess = KGE(sim, obs)
    safe = lambda v: -99999.0 if np.isnan(v) else float(v)
    return {
        "NSE": float(NS(sim, obs)), "KGE": safe(kge), "KGE-A": float(a),
        "KGE-B": float(b), "Corr": safe(corr),
        "mse": float(mean_squared_error(obs, sim)), "KGEss": safe(kgess),
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

def save_series(path, tensor, time, phase, prefix):
    a = tensor.detach().cpu().numpy()
    if a.shape[1] == 1:
        df = pd.DataFrame({"time": time, prefix: a[:, 0], "phase": phase})
    else:
        df = pd.DataFrame(a, columns=[f"{prefix}_{i+1}" for i in range(a.shape[1])])
        df.insert(0, "time", time)
        df["phase"] = phase
    df.to_csv(path, index=False)

class Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.MCPBRNNNode = MCPNet_DI_SAO_SingleLayer(hidden_size)

    def forward(self, x, c_mean, c_std):
        return self.MCPBRNNNode(x, c_mean, c_std)

def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint, args.hidden_size)
    data_dir = resolve_path(args.data_dir, PROJECT_DIR / "20220527-MDUPLEX-LeafRiver")
    out_dir = resolve_path(args.output_dir, SCRIPT_DIR / f"{args.hidden_size}nodes" / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, masks = load_data(data_dir, device)
    model = Model(args.hidden_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()

    with torch.no_grad():
        result = model(x, args.c_mean, args.c_std)

    row = {}
    for split in SPLITS:
        sim = torch.masked_select(result[0], masks[split]).unsqueeze(1)
        obs = torch.masked_select(y, masks[split]).unsqueeze(1)
        suffix = "" if split == "train" else f"_{split}"
        for name, value in calc_metrics(sim, obs).items():
            row[f"{name}{suffix}"] = value

    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)

    print("\nSkill metrics")
    for split in SPLITS:
        suffix = "" if split == "train" else f"_{split}"
        print(f"{split:9s} | " + " | ".join(f"{m}={row[f'{m}{suffix}']:.4f}" for m in METRICS))

    time, phase = build_time_axis(x.shape[0])
    outputs = [
        ("discharge.csv", result[0], "discharge"),
        ("node_discharge.csv", result[1], "node_discharge"),
        ("storage.csv", result[2], "storage"),
        ("loss_unconstrained.csv", result[3], "loss_unconstrained"),
        ("loss_constrained.csv", result[4], "loss_constrained"),
        ("input_bypass.csv", result[5], "input_bypass"),
        ("input_gate.csv", result[6], "input_gate"),
        ("output_gate.csv", result[7], "output_gate"),
        ("loss_gate.csv", result[8], "loss_gate"),
        ("loss_gate_constrained.csv", result[9], "loss_gate_constrained"),
        ("remember_gate.csv", result[10], "remember_gate"),
    ]
    for name, tensor, prefix in outputs:
        save_series(out_dir / name, tensor, time, phase, prefix)

    print(f"\nLoaded checkpoint: {checkpoint}")
    print(f"Saved outputs to: {out_dir}")

if __name__ == "__main__":
    main()
