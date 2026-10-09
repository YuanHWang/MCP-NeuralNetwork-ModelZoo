import torch
import torch.nn as nn


__all__ = [
    "MC_Linear_Layer",
    "MR_Linear_Layer",
    "ML_Linear_Layer",
]


class MC_Linear_Layer(nn.Module):
    """Mass-conserving inter-layer mapping."""

    def __init__(
        self,
        input_size: int,
        hidden_size_old: int,
        hidden_size_new: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_old = hidden_size_old
        self.hidden_size_new = hidden_size_new

        self.weight_w = nn.Parameter(
            torch.empty(
                hidden_size_old,
                hidden_size_new,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_w.uniform_(
                0.0,
                1.0,
            )

    def forward(self, x):
        weight = torch.softmax(
            self.weight_w,
            dim=0,
        )

        return torch.mm(
            x,
            weight,
        )


class MR_Linear_Layer(nn.Module):
    """Positive, non-normalized mass-relaxed inter-layer mapping."""

    def __init__(
        self,
        input_size: int,
        hidden_size_old: int,
        hidden_size_new: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_old = hidden_size_old
        self.hidden_size_new = hidden_size_new

        self.weight_w = nn.Parameter(
            torch.empty(
                hidden_size_old,
                hidden_size_new,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_w.uniform_(
                0.0,
                1.0,
            )

    def forward(self, x):
        weight = torch.exp(
            self.weight_w
        )

        return torch.mm(
            x,
            weight,
        )


class ML_Linear_Layer(nn.Module):
    """Unconstrained affine inter-layer mapping."""

    def __init__(
        self,
        input_size: int,
        hidden_size_old: int,
        hidden_size_new: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_old = hidden_size_old
        self.hidden_size_new = hidden_size_new

        self.weight_w = nn.Parameter(
            torch.empty(
                hidden_size_old,
                hidden_size_new,
            )
        )
        self.bias = nn.Parameter(
            torch.empty(1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_w.uniform_(
                0.0,
                1.0,
            )
            self.bias.uniform_(
                0.0,
                1.0,
            )

    def forward(self, x):
        bias = self.bias.unsqueeze(0).expand(
            1,
            self.hidden_size_new,
        )

        return torch.addmm(
            bias,
            x,
            self.weight_w,
        )
