import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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

from MCPBRNN_lib_tools.Eval_Metric import KGE
from MCPBRNN_lib_tools.Loss_Function import KGELoss
from MCPBRNN_lib_tools.miscellaneous import (
    MCPBRNN_Generic_PETconstraint_LossMultiInput,
)


SPIN_LEN = 1095
TRAIN_TIME_LEN = 8400
TIME_LEN = 15705

# Historical transfer from the older single-node Scaling model.
TRANSFER_KEYS = (
    "weight_r_yom",
    "weight_r_ylm",
    "weight_r_yfm",
    "bias_b0_yom",
    "weight_b1_yom",
    "bias_b0_ylm",
    "weight_b2_ylm",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Miscellaneous/SingleNode-MN model."
    )
    parser.add_argument("--epoch_no", type=int, default=3000)
    parser.add_argument("--seed_no", type=int, default=2925)
    parser.add_argument("--c_mean", type=float, default=412.9139085)
    parser.add_argument("--c_std", type=float, default=77.49943867)
    parser.add_argument("--time_lag", type=int, default=0)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--initial_checkpoint",
        default=None,
        help=(
            "Optional historical Scaling-model checkpoint. "
            "When supplied, the seven parameters used in the original "
            "training script are transferred into the new model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: ./training",
    )
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
    return x, y, masks


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.MCPBRNNNode = MCPBRNN_Generic_PETconstraint_LossMultiInput(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            spinLen=SPIN_LEN,
            traintimeLen=TRAIN_TIME_LEN,
            initial_forget_bias=0,
        )

    def forward(self, x, epoch, time_lag, y, c_mean, c_std):
        result = self.MCPBRNNNode(
            x,
            epoch,
            time_lag,
            y,
            c_mean,
            c_std,
        )
        return result[0]


def normal_initialize(model):
    """Preserve the historical N(0,1) initialization used by the script."""
    with torch.no_grad():
        for parameter in model.parameters():
            values = np.random.normal(
                0.0,
                1.0,
                size=tuple(parameter.shape),
            )
            parameter.copy_(
                torch.tensor(
                    values,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )


def load_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def transfer_historical_initialization(model, checkpoint, device):
    """Transfer exactly the seven parameters copied in the original script."""
    state = load_state(checkpoint, device)

    with torch.no_grad():
        for name in TRANSFER_KEYS:
            target_key = f"MCPBRNNNode.{name}"

            if target_key in state:
                source = state[target_key]
            elif name in state:
                source = state[name]
            else:
                raise KeyError(
                    f"Initial checkpoint does not contain {target_key} or {name}."
                )

            target = getattr(model.MCPBRNNNode, name)
            target.copy_(
                source.to(
                    dtype=target.dtype,
                    device=target.device,
                ).reshape_as(target)
            )


def kge_value(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()
    return float(KGE(sim, obs)[0])


def main():
    args = parse_args()

    np.random.seed(args.seed_no)
    torch.manual_seed(args.seed_no)

    device = resolve_device(args.device)
    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )
    output_dir = resolve_path(
        args.output_dir,
        SCRIPT_DIR / "training",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, masks = load_data(data_dir, device)

    model = Model().to(device)
    normal_initialize(model)

    if args.initial_checkpoint:
        initial_checkpoint = resolve_path(
            args.initial_checkpoint,
            SCRIPT_DIR,
        )
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {initial_checkpoint}"
            )
        transfer_historical_initialization(
            model,
            initial_checkpoint,
            device,
        )
        print(f"Transferred historical initialization from: {initial_checkpoint}")
    else:
        print(
            "No --initial_checkpoint supplied; training starts from the "
            "historical N(0,1) random initialization."
        )

    loss_func = KGELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.025,
    )
    learning_rates = {
        300: 0.0125,
        600: 0.0125,
    }

    history = []

    for epoch in range(1, args.epoch_no + 1):
        if epoch in learning_rates:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch]

        model.train()
        optimizer.zero_grad()

        prediction = model(
            x,
            epoch,
            args.time_lag,
            y,
            args.c_mean,
            args.c_std,
        )

        sim_train = torch.masked_select(
            prediction,
            masks["train"],
        ).unsqueeze(1)
        obs_train = torch.masked_select(
            y,
            masks["train"],
        ).unsqueeze(1)

        loss = loss_func(sim_train, obs_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction = model(
                x,
                epoch,
                args.time_lag,
                y,
                args.c_mean,
                args.c_std,
            )

            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "loss": float(loss.detach().cpu()),
            }

            for split, mask in masks.items():
                sim = torch.masked_select(
                    prediction,
                    mask,
                ).unsqueeze(1)
                obs = torch.masked_select(
                    y,
                    mask,
                ).unsqueeze(1)
                row[f"KGE_{split}"] = kge_value(sim, obs)

        history.append(row)

        # Keep one checkpoint only: overwrite the current training checkpoint.
        torch.save(
            model.state_dict(),
            output_dir / "model.pt",
        )

        print(
            f"Epoch {epoch:4d} | "
            f"train KGE={row['KGE_train']:.4f} | "
            f"selection KGE={row['KGE_selection']:.4f}"
        )

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    print(f"\nSaved checkpoint: {output_dir / 'model.pt'}")
    print(f"Saved history:    {output_dir / 'training_history.csv'}")


if __name__ == "__main__":
    main()
