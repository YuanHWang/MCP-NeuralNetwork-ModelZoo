import torch
import torch.nn as nn


__all__ = [
    "MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_LossGateUpdate",
    "MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_shareoutputlossGate_LossGateUpdate",
]


_PET_MEAN = 2.9086
_PET_STD = 1.8980


def _prepare_input(x, batch_first):
    """Return input as [seq, batch, features] and validate the pruning setup."""
    if x.ndim != 3:
        raise ValueError(
            "Expected a 3D tensor [batch, seq, features] or "
            f"[seq, batch, features], got shape {tuple(x.shape)}."
        )

    if batch_first:
        x = x.transpose(0, 1)

    seq_len, batch_size, n_features = x.shape

    if seq_len != 1:
        raise ValueError(
            "The Paper-3 pruning cases use seq_length == 1."
        )

    if n_features < 2:
        raise ValueError(
            "The pruning MCP requires precipitation and PET."
        )

    return x, batch_size


def _pet_constrained_fraction(
    loss_fraction,
    pet,
    storage,
    relu,
):
    """Apply the original PET constraint without explicit scalar loops."""
    positive_storage = storage > 0

    safe_storage = torch.where(
        positive_storage,
        storage,
        torch.ones_like(storage),
    )

    constrained = (
        loss_fraction
        - relu(
            loss_fraction
            - pet / safe_storage
        )
    )

    return torch.where(
        positive_storage,
        constrained,
        loss_fraction,
    )


def _run_pruning_unit(
    module,
    x,
    time_lag,
    y_obs,
    cmean,
    cstd,
    *,
    shared_output_loss,
):
    """Core recurrence shared by the two pruning classes."""
    if cstd == 0:
        raise ValueError("cstd must be non-zero.")

    x, batch_size = _prepare_input(
        x,
        module.batch_first,
    )

    if not 0 <= time_lag <= batch_size:
        raise ValueError(
            f"time_lag must be between 0 and {batch_size}, got {time_lag}."
        )

    hidden_size = module.hidden_size
    storage = x.new_zeros(1, hidden_size)

    discharge = x.new_zeros(
        batch_size,
        hidden_size,
    )
    storage_series = x.new_zeros(
        batch_size,
        hidden_size,
    )
    loss_unconstrained = x.new_zeros(
        batch_size,
        hidden_size,
    )
    loss_constrained = x.new_zeros(
        batch_size,
        hidden_size,
    )
    bypass = x.new_zeros(
        batch_size,
        hidden_size,
    )

    gate_input = x.new_zeros(
        batch_size,
        hidden_size,
    )
    gate_output = x.new_zeros(
        batch_size,
        hidden_size,
    )
    gate_loss = x.new_zeros(
        batch_size,
        hidden_size,
    )
    gate_loss_constrained = x.new_zeros(
        batch_size,
        hidden_size,
    )
    gate_remember = x.new_zeros(
        batch_size,
        hidden_size,
    )

    # Kept only because the current pruning evaluation wrappers
    # unpack this legacy diagnostic.
    obs_std = x.new_zeros(
        batch_size,
        hidden_size,
    )
    observed_std = torch.std(
        y_obs[
            module.spinLen:
            module.traintimeLen
        ]
    )

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

    for b in range(
        time_lag,
        batch_size,
    ):
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

        storage_scaled = (
            storage - cmean
        ) / cstd

        if shared_output_loss:
            output_state = torch.addmm(
                output_bias,
                storage_scaled,
                module.weight_b1_yom,
            )
            loss_storage_state = torch.mm(
                storage_scaled,
                module.weight_b1_ylm,
            )
        else:
            output_state = (
                output_bias
                + storage_scaled
                * module.weight_b1_yom
            )
            loss_storage_state = (
                storage_scaled
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
            + (
                (pet - _PET_MEAN)
                / _PET_STD
            )
            * module.weight_b2_ylm
        )

        loss_fraction = (
            loss_scale
            * torch.sigmoid(
                loss_storage_state
                + loss_pet_state
            )
        )

        constrained_loss_fraction = (
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
            - constrained_loss_fraction
        )

        # Preserve original PRE-update timing.
        storage_series[b, :] = storage[0]
        discharge[b, :] = (
            output_fraction * storage
        )[0]
        loss_unconstrained[b, :] = (
            loss_fraction * storage
        )[0]
        loss_constrained[b, :] = (
            constrained_loss_fraction
            * storage
        )[0]

        # Input bypass is identically zero in these pruning cases.
        gate_output[b, :] = (
            output_fraction[0]
        )
        gate_loss[b, :] = (
            loss_fraction[0]
        )
        gate_loss_constrained[b, :] = (
            constrained_loss_fraction[0]
        )
        gate_remember[b, :] = (
            remember_fraction[0]
        )

        obs_std[b, :] = observed_std

        storage = (
            remember_fraction * storage
            + precipitation
        )

    output_weights = torch.softmax(
        module.wo[:, 0],
        dim=0,
    ).unsqueeze(1)

    total_discharge = torch.mm(
        discharge,
        output_weights,
    )

    return (
        total_discharge,
        discharge,
        storage_series,
        loss_unconstrained,
        loss_constrained,
        bypass,
        gate_input,
        gate_output,
        gate_loss,
        gate_loss_constrained,
        gate_remember,
        obs_std,
    )


class MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_LossGateUpdate(
    nn.Module
):
    """Pruning Case 1: five independent MCP nodes with mass-conserved output."""

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        gate_dim_ucorr: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retain the legacy constructor for drop-in compatibility.
        del input_size, gate_dim, gate_dim_ucorr
        del initial_forget_bias

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        self.weight_r_yom = nn.Parameter(
            torch.empty(1, hidden_size)
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(1, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(1, hidden_size)
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(1, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(1, hidden_size)
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(1, hidden_size)
        )

        self.wo = nn.Parameter(
            torch.empty(hidden_size, 1)
        )

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            # Preserve the original draw order.
            self.wo.uniform_(0.0, 1.0)
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        y_obs,
        cmean,
        cstd,
    ):
        del epoch

        return _run_pruning_unit(
            self,
            x,
            time_lag,
            y_obs,
            cmean,
            cstd,
            shared_output_loss=False,
        )


class MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_shareoutputlossGate_LossGateUpdate(
    nn.Module
):
    """Pruning Case 2: MCP nodes share storage information in output/loss gates."""

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        gate_dim_ucorr: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retain the legacy constructor for drop-in compatibility.
        del input_size, gate_dim, gate_dim_ucorr
        del initial_forget_bias

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        self.weight_r_yom = nn.Parameter(
            torch.empty(1, hidden_size)
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(1, hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(1, hidden_size)
        )

        self.bias_b0_yom = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(
                hidden_size,
                hidden_size,
            )
        )

        self.bias_b0_ylm = nn.Parameter(
            torch.empty(hidden_size)
        )
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(
                hidden_size,
                hidden_size,
            )
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(1, hidden_size)
        )

        self.wo = nn.Parameter(
            torch.empty(hidden_size, 1)
        )

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            # Preserve the original draw order.
            self.wo.uniform_(0.0, 1.0)
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        epoch,
        time_lag,
        y_obs,
        cmean,
        cstd,
    ):
        del epoch

        return _run_pruning_unit(
            self,
            x,
            time_lag,
            y_obs,
            cmean,
            cstd,
            shared_output_loss=True,
        )
