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
        f"Could not find project root containing MCPBRNN_lib_tools above {start}"
    )


PROJECT_DIR = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS
from MCPBRNN_lib_tools.miscellaneous import (
    MCPBRNN_Generic_PETconstraint_LossMultiInput_DI,
)


SPIN_LEN = 1095
TRAIN_TIME_LEN = 8400
TIME_LEN = 15705
METRICS = ("NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss")
SPLITS = ("train", "selection", "testing", "spinup")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Miscellaneous/SingleNode-DI."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--c_mean", type=float, default=412.9139085)
    parser.add_argument("--c_std", type=float, default=77.49943867)
    parser.add_argument("--time_lag", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


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

    if len(data) != TIME_LEN or len(flags) != TIME_LEN:
        raise ValueError(
            f"Expected {TIME_LEN} daily records, "
            f"found data={len(data)}, flags={len(flags)}."
        )

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


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.MCPBRNNNode = MCPBRNN_Generic_PETconstraint_LossMultiInput_DI(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            spinLen=SPIN_LEN,
            traintimeLen=TRAIN_TIME_LEN,
            initial_forget_bias=0,
        )

    def forward(self, x, y, c_mean, c_std, time_lag=0):
        result = self.MCPBRNNNode(
            x,
            0,
            time_lag,
            y,
            c_mean,
            c_std,
        )
        return result


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


def save_series(path, tensor, names):
    values = tensor.detach().cpu().numpy()
    if values.ndim == 1:
        values = values[:, None]
    pd.DataFrame(values, columns=names).to_csv(path, index=False)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint)

    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )
    output_dir = resolve_path(
        args.output_dir,
        SCRIPT_DIR / "evaluation",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data, flags, x, y, masks = load_data(data_dir, device)

    model = Model().to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)

    model.eval()
    with torch.no_grad():
        result = model(
            x,
            y,
            args.c_mean,
            args.c_std,
            args.time_lag,
        )

    # DI historical return:
    # 0 hnout, 1 h_n, 2 c_n, 3 l_n, 4 lc_n, ...
    prediction = result[0]

    rows = []
    for split in SPLITS:
        sim = torch.masked_select(prediction, masks[split]).unsqueeze(1)
        obs = torch.masked_select(y, masks[split]).unsqueeze(1)
        row = {"split": split, "n": int(sim.numel())}
        row.update(calculate_metrics(sim, obs))
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    print("\nSkill metrics")
    for _, row in metrics.iterrows():
        print(
            f"{row['split']:9s} | "
            + " | ".join(
                f"{metric}={row[metric]:.4f}"
                for metric in METRICS
            )
        )

    save_series(output_dir / "discharge.csv", result[1], ["Qsim"])
    save_series(output_dir / "storage.csv", result[2], ["storage"])
    save_series(output_dir / "loss_unconstrained.csv", result[3], ["loss"])
    save_series(output_dir / "loss_constrained.csv", result[4], ["loss_constrained"])
    save_series(output_dir / "input_bypass.csv", result[5], ["input_bypass"])
    save_series(output_dir / "input_gate.csv", result[6], ["input_gate"])
    save_series(output_dir / "output_gate.csv", result[7], ["output_gate"])
    save_series(output_dir / "loss_gate.csv", result[8], ["loss_gate"])
    save_series(output_dir / "loss_gate_constrained.csv", result[9], ["loss_gate_constrained"])
    save_series(output_dir / "remember_gate.csv", result[10], ["remember_gate"])

    pd.DataFrame(
        {
            "parameter": ["wi", "exp(wi)"],
            "value": [
                float(model.MCPBRNNNode.wi.detach().cpu().reshape(-1)[0]),
                float(torch.exp(model.MCPBRNNNode.wi).detach().cpu().reshape(-1)[0]),
            ],
        }
    ).to_csv(output_dir / "input_weight.csv", index=False)

    forcing = data.copy()
    forcing["skill_flag"] = flags.to_numpy()
    forcing.to_csv(output_dir / "forcing_observation.csv", index=False)

    print(
        "\nLearned input weight | "
        f"wi={model.MCPBRNNNode.wi.item():.6f} | "
        f"exp(wi)={torch.exp(model.MCPBRNNNode.wi).item():.6f}"
    )
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
