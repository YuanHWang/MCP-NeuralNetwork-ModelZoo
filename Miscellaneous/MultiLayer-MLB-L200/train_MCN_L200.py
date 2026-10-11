import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

ARCH = "L200"
HIDDEN_SIZE = 2
MODEL_LABEL = "non-sharing parallel-node"

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
from MCPBRNN_lib_tools.Loss_Function import KGELoss
from MCPBRNN_lib_tools.MCNZoo_MultiLayers import MCN_unit
from MCPBRNN_lib_tools.miscellaneous import MCN_SingleMultiML_IC


METRICS = ["NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss"]
SPLITS = ("train", "selection", "testing", "spinup")


def parse_args():
    p = argparse.ArgumentParser(
        description=f"Train {MODEL_LABEL} MCP architecture {ARCH}."
    )
    p.add_argument("--case_no", type=int, default=0)
    p.add_argument("--epoch_no", type=int, default=3000)
    p.add_argument("--seed_no", type=int, default=2925)
    p.add_argument("--c_mean", type=float, default=412.9139085)
    p.add_argument("--c_std", type=float, default=77.49943867)
    p.add_argument(
        "--init_checkpoint",
        default=None,
        help=(
            "Single-node checkpoint used for historical initialization. "
            "If omitted, the script looks in Miscellaneous/SingleNode-MN."
        ),
    )
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return p.parse_args()


def resolve_initial_checkpoint(value):
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = (SCRIPT_DIR / path).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {path}"
            )

        return path

    single_node_dir = PROJECT_DIR / "Miscellaneous" / "SingleNode-MN"

    candidates = sorted(single_node_dir.glob("best_model_epoch*.pt"))

    if not candidates:
        candidates = sorted(single_node_dir.glob("*.pt"))

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not determine the single-node initialization checkpoint. "
            f"Expected exactly one .pt file in {single_node_dir}, "
            f"but found {len(candidates)}. "
            "Use --init_checkpoint to specify it explicitly."
        )

    return candidates[0]


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

        self.MCPBRNNNode = MCN_SingleMultiML_IC(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            initial_forget_bias=0,
        )

    def forward(self, x, c_mean, c_std):
        return self.MCPBRNNNode(x, c_mean, c_std)


class SingleNodeModel(nn.Module):
    """Single-node MCP used only to reproduce the historical initialization."""

    def __init__(self):
        super().__init__()

        self.MCPBRNNNode = MCN_unit(
            input_size=1,
            hidden_size=1,
            initial_forget_bias=0,
        )


def initialize_from_single_node(model, checkpoint, device):
    source = SingleNodeModel().to(device)

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    source.load_state_dict(state, strict=True)

    src = source.MCPBRNNNode
    dst = model.MCPBRNNNode

    def scalar(parameter):
        return float(parameter.detach().reshape(-1)[0].cpu())

    with torch.no_grad():
        for i in range(HIDDEN_SIZE):
            value = scalar(src.weight_r_yom)
            dst.weight_r_yom[0, i] = np.random.normal(
                value, 0.10 * abs(value)
            )

            value = scalar(src.weight_r_ylm)
            dst.weight_r_ylm[0, i] = np.random.normal(
                value, 0.10 * abs(value)
            )

            value = scalar(src.weight_r_yfm)
            dst.weight_r_yfm[0, i] = np.random.normal(
                value, 0.025 * abs(value)
            )

            value = scalar(src.bias_b0_yom)
            dst.bias_b0_yom[i] = np.random.normal(
                value, 0.20 * abs(value)
            )

            value = scalar(src.weight_b1_yom)
            dst.weight_b1_yom[0, i] = np.random.normal(
                value, 0.10 * abs(value)
            )

            value = scalar(src.bias_b0_ylm)
            dst.bias_b0_ylm[i] = np.random.normal(
                value, 0.05 * abs(value)
            )

            value = scalar(src.weight_b1_ylm)
            dst.weight_b1_ylm[0, i] = np.random.normal(
                value, 0.05 * abs(value)
            )

            value = scalar(src.weight_b2_ylm)
            dst.weight_b2_ylm[0, i] = np.random.normal(
                value, 0.05 * abs(value)
            )

            output_weight = 1.0 / HIDDEN_SIZE
            dst.wo[i, 0] = np.random.normal(
                output_weight,
                0.05 * abs(output_weight),
            )

        dst.bias[0, 0] = np.random.normal(0.0, 0.01)


def main():
    args = parse_args()

    np.random.seed(args.seed_no)
    torch.manual_seed(args.seed_no)

    device = resolve_device(args.device)

    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )

    out_dir = resolve_path(
        args.output_dir,
        SCRIPT_DIR / f"case_{args.case_no}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    init_checkpoint = resolve_initial_checkpoint(args.init_checkpoint)

    x, y, masks = load_data(data_dir, device)

    model = Model().to(device)
    initialize_from_single_node(
        model,
        init_checkpoint,
        device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.025,
    )
    loss_func = KGELoss()

    history = []
    best_epoch = None
    best_kge = -np.inf
    best_state = None

    for epoch in range(1, args.epoch_no + 1):
        if epoch in (300, 600):
            for group in optimizer.param_groups:
                group["lr"] = 0.0125

        model.train()
        optimizer.zero_grad()

        q = model(x, args.c_mean, args.c_std)[0]

        sim = torch.masked_select(
            q,
            masks["train"],
        ).unsqueeze(1)

        obs = torch.masked_select(
            y,
            masks["train"],
        ).unsqueeze(1)

        loss = loss_func(sim, obs)
        loss.backward()
        optimizer.step()

        _, row = evaluate(
            model,
            x,
            y,
            masks,
            args.c_mean,
            args.c_std,
        )

        row["epoch"] = epoch
        history.append(row)

        if row["KGE_selection"] > best_kge:
            best_kge = row["KGE_selection"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"{ARCH} | Epoch {epoch:4d} | "
            f"KGE train={row['KGE']:.4f} | "
            f"selection={row['KGE_selection']:.4f} | "
            f"test={row['KGE_testing']:.4f}"
        )

    columns = ["epoch"] + METRICS + [
        f"{metric}_{split}"
        for split in ("selection", "testing", "spinup")
        for metric in METRICS
    ]

    pd.DataFrame(history)[columns].to_csv(
        out_dir / "metrics_by_epoch.csv",
        index=False,
    )

    if best_state is None:
        raise RuntimeError("No best model was selected.")

    best_file = out_dir / f"best_model_epoch{best_epoch}.pt"
    torch.save(best_state, best_file)

    model.load_state_dict(best_state, strict=True)

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
    print(f"Initialization checkpoint: {init_checkpoint}")
    print(f"Saved best model: {best_file}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
