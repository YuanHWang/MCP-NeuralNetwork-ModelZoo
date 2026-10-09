import torch
import torch.nn as nn


__all__ = [
    "MCN_unit",
    "MCN_unit_sharedlossoutput",
]


_PET_MEAN = 2.9086
_PET_STD = 1.8980


def _prepare_input(x, batch_first):
    """Return input as [seq, batch, features] and validate the MCP interface."""
    if x.ndim != 3:
        raise ValueError(
            "Expected a 3D tensor [batch, seq, features] or "
            f"[seq, batch, features], got {tuple(x.shape)}."
        )

    if batch_first:
        x = x.transpose(0, 1)

    seq_len, batch_size, n_features = x.shape

    if seq_len != 1:
        raise ValueError(
            "Multi-layer MCP units require seq_length == 1."
        )

    if n_features < 2:
        raise ValueError(
            "Multi-layer MCP units require at least two input features."
        )

    return x, batch_size


def _pet_constrained_fraction(
    loss_fraction,
    pet,
    storage,
    relu,
):
    """Apply the original PET constraint without dividing by zero."""
    positive_storage = storage > 0

    safe_storage = torch.where(
        positive_storage,
        storage,
        torch.ones_like(storage),
    )

    constrained = loss_fraction - relu(
        loss_fraction - pet / safe_storage
    )

    return torch.where(
        positive_storage,
        constrained,
        loss_fraction,
    )


def _run_unit(
    module,
    x,
    cmean,
    cstd,
    *,
    shared,
):
    """Shared implementation for the two MCP units used by the six families."""
    if cstd == 0:
        raise ValueError("cstd must be non-zero.")

    x, batch_size = _prepare_input(
        x,
        module.batch_first,
    )

    hidden_size = module.hidden_size
    storage = x.new_zeros(1, hidden_size)

    exp_output = torch.exp(
        module.weight_r_yom
    )
    exp_loss = torch.exp(
        module.weight_r_ylm
    )
    exp_remember = torch.exp(
        module.weight_r_yfm
    )

    denominator = (
        exp_output
        + exp_loss
        + exp_remember
    )

    output_scale = (
        exp_output / denominator
    )
    loss_scale = (
        exp_loss / denominator
    )

    output_bias = (
        module.bias_b0_yom.unsqueeze(0)
    )
    loss_bias = (
        module.bias_b0_ylm.unsqueeze(0)
    )

    discharge_series = []
    storage_series = []
    loss_series = []
    constrained_loss_series = []
    bypass_series = []

    input_gate_series = []
    output_gate_series = []
    loss_gate_series = []
    constrained_loss_gate_series = []
    remember_gate_series = []

    for b in range(batch_size):
        # Preserve the original feature convention:
        # feature 0 drives inflow; feature 1 drives the loss gate.
        precipitation = (
            x[0, b, 0]
            .reshape(1, 1)
            .expand(1, hidden_size)
        )
        pet = (
            x[0, b, 1]
            .reshape(1, 1)
            .expand(1, hidden_size)
        )

        standardized_storage = (
            storage - cmean
        ) / cstd

        if shared:
            output_state = torch.addmm(
                output_bias,
                standardized_storage,
                module.weight_b1_yom,
            )

            loss_storage_state = torch.mm(
                standardized_storage,
                module.weight_b1_ylm,
            )
        else:
            output_state = (
                output_bias
                + standardized_storage
                * module.weight_b1_yom
            )

            loss_storage_state = (
                standardized_storage
                * module.weight_b1_ylm
            )

        output_fraction = (
            output_scale
            * torch.sigmoid(
                output_state
            )
        )

        loss_pet_state = (
            loss_bias
            + ((pet - _PET_MEAN) / _PET_STD)
            * module.weight_b2_ylm
        )

        loss_fraction = (
            loss_scale
            * torch.sigmoid(
                loss_storage_state
                + loss_pet_state
            )
        )

        loss_fraction_constrained = (
            _pet_constrained_fraction(
                loss_fraction,
                pet,
                storage,
                module.relu_l,
            )
        )

        remember_fraction = (
            1.0
            - output_fraction
            - loss_fraction_constrained
        )

        zero = torch.zeros_like(
            storage
        )

        # Preserve the original PRE-update timing.
        storage_series.append(
            storage
        )
        discharge_series.append(
            output_fraction * storage
        )
        loss_series.append(
            loss_fraction * storage
        )
        constrained_loss_series.append(
            loss_fraction_constrained
            * storage
        )
        bypass_series.append(
            zero
        )

        input_gate_series.append(
            zero
        )
        output_gate_series.append(
            output_fraction
        )
        loss_gate_series.append(
            loss_fraction
        )
        constrained_loss_gate_series.append(
            loss_fraction_constrained
        )
        remember_gate_series.append(
            remember_fraction
        )

        storage = (
            remember_fraction * storage
            + precipitation
        )

    return (
        torch.cat(
            discharge_series,
            dim=0,
        ),
        torch.cat(
            storage_series,
            dim=0,
        ),
        torch.cat(
            loss_series,
            dim=0,
        ),
        torch.cat(
            constrained_loss_series,
            dim=0,
        ),
        torch.cat(
            bypass_series,
            dim=0,
        ),
        torch.cat(
            input_gate_series,
            dim=0,
        ),
        torch.cat(
            output_gate_series,
            dim=0,
        ),
        torch.cat(
            loss_gate_series,
            dim=0,
        ),
        torch.cat(
            constrained_loss_gate_series,
            dim=0,
        ),
        torch.cat(
            remember_gate_series,
            dim=0,
        ),
    )


class MCN_unit(nn.Module):
    """MCP unit used by the three non-InfoShared multi-layer families."""

    def __init__(
        self,
        input_size: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = (
            initial_forget_bias
        )

        self.weight_r_yom = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(
                hidden_size
            )
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )

        self.bias_b0_ylm = nn.Parameter(
            torch.empty(
                hidden_size
            )
        )
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(
                0.0,
                1.0,
            )
            self.weight_r_ylm.uniform_(
                0.0,
                1.0,
            )
            self.weight_r_yfm.uniform_(
                0.0,
                1.0,
            )
            self.bias_b0_yom.uniform_(
                0.0,
                1.0,
            )
            self.weight_b1_yom.uniform_(
                0.0,
                1.0,
            )
            self.bias_b0_ylm.uniform_(
                0.0,
                1.0,
            )
            self.weight_b1_ylm.uniform_(
                0.0,
                1.0,
            )
            self.weight_b2_ylm.uniform_(
                0.0,
                1.0,
            )

    def forward(
        self,
        x,
        cmean,
        cstd,
    ):
        return _run_unit(
            self,
            x,
            cmean,
            cstd,
            shared=False,
        )


class MCN_unit_sharedlossoutput(nn.Module):
    """MCP unit used by the three InfoShared multi-layer families."""

    def __init__(
        self,
        input_size: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = (
            initial_forget_bias
        )

        self.weight_r_yom = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(
                hidden_size
            )
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(
                hidden_size,
                hidden_size,
            )
        )

        self.bias_b0_ylm = nn.Parameter(
            torch.empty(
                hidden_size
            )
        )
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(
                hidden_size,
                hidden_size,
            )
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(
                1,
                hidden_size,
            )
        )

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(
                0.0,
                1.0,
            )
            self.weight_r_ylm.uniform_(
                0.0,
                1.0,
            )
            self.weight_r_yfm.uniform_(
                0.0,
                1.0,
            )
            self.bias_b0_yom.uniform_(
                0.0,
                1.0,
            )
            self.weight_b1_yom.uniform_(
                0.0,
                1.0,
            )
            self.bias_b0_ylm.uniform_(
                0.0,
                1.0,
            )
            self.weight_b1_ylm.uniform_(
                0.0,
                1.0,
            )
            self.weight_b2_ylm.uniform_(
                0.0,
                1.0,
            )

    def forward(
        self,
        x,
        cmean,
        cstd,
    ):
        return _run_unit(
            self,
            x,
            cmean,
            cstd,
            shared=True,
        )
