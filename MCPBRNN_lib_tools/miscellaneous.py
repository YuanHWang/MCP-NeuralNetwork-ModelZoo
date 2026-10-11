import torch
import torch.nn as nn


__all__ = [
    "MCPBRNN_Generic_PETconstraint_LossMultiInput",
    "MCPBRNN_Generic_PETconstraint_LossMultiInput_DI",
    "MCPBRNN_Generic_PETconstraint_LossMultiInput_DS",
    "MCPBRNN_Generic_PETconstraint_LossMultiInput_MLB",
    "MCN_SingleMultiML_SharedLossOutput_IC",
    "MCN_SingleMultiML_IC",
]


_PET_MEAN = 2.9086
_PET_STD = 1.8980


def _validate_input(x, batch_first):
    if x.ndim != 3:
        raise ValueError(
            "Expected x with shape [batch, seq, features] "
            "or [seq, batch, features]."
        )

    if batch_first:
        x = x.transpose(0, 1)

    seq_len, batch_size, n_features = x.shape

    if seq_len != 1:
        raise ValueError(
            "These historical miscellaneous cases use seq_length == 1."
        )
    if n_features < 2:
        raise ValueError("The model requires precipitation and PET.")

    return x, batch_size


def _loss_constraint(loss_fraction, pet, storage, relu):
    positive_storage = storage > 0
    safe_storage = torch.where(
        positive_storage,
        storage,
        torch.ones_like(storage),
    )
    constrained = (
        loss_fraction
        - relu(loss_fraction - pet / safe_storage)
    )
    return torch.where(
        positive_storage,
        constrained,
        loss_fraction,
    )


class _LossMultiInputBase(nn.Module):
    """Shared recurrence for the historical single-node miscellaneous cases."""

    def _initialize_common_parameters(self):
        self.weight_r_yom = nn.Parameter(
            torch.empty(self.hidden_size, self.hidden_size)
        )
        self.weight_r_ylm = nn.Parameter(
            torch.empty(self.hidden_size, self.hidden_size)
        )
        self.weight_r_yfm = nn.Parameter(
            torch.empty(self.hidden_size, self.hidden_size)
        )
        self.bias_b0_yom = nn.Parameter(
            torch.empty(self.hidden_size)
        )
        self.weight_b1_yom = nn.Parameter(
            torch.empty(self.hidden_size, self.hidden_size)
        )
        self.bias_b0_ylm = nn.Parameter(
            torch.empty(self.hidden_size)
        )
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(self.hidden_size, self.hidden_size)
        )
        self.weight_b2_ylm = nn.Parameter(
            torch.empty(self.hidden_size, self.hidden_size)
        )
        self.relu_l = nn.ReLU()

    def _reset_common_parameters(self):
        self.weight_r_yom.uniform_(0.0, 1.0)
        self.weight_r_ylm.uniform_(0.0, 1.0)
        self.weight_r_yfm.uniform_(0.0, 1.0)
        self.bias_b0_yom.uniform_(0.0, 1.0)
        self.weight_b1_yom.uniform_(0.0, 1.0)
        self.bias_b0_ylm.uniform_(0.0, 1.0)
        self.weight_b1_ylm.uniform_(0.0, 1.0)
        self.weight_b2_ylm.uniform_(0.0, 1.0)

    def _run_core(
        self,
        x,
        time_lag,
        y_obs,
        cmean,
        cstd,
        precipitation_scale=None,
    ):
        if cstd == 0:
            raise ValueError("cstd must be non-zero.")

        x, batch_size = _validate_input(x, self.batch_first)

        if not 0 <= time_lag <= batch_size:
            raise ValueError(
                f"time_lag must be between 0 and {batch_size}, got {time_lag}."
            )

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        l_n = x.new_zeros(batch_size, hidden_size)
        lc_n = x.new_zeros(batch_size, hidden_size)
        bp_n = x.new_zeros(batch_size, hidden_size)
        gate_ib = x.new_zeros(batch_size, hidden_size)
        gate_oo = x.new_zeros(batch_size, hidden_size)
        gate_ol = x.new_zeros(batch_size, hidden_size)
        gate_olc = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)
        obs_std = x.new_zeros(batch_size, hidden_size)

        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        obsstd = torch.std(
            y_obs[self.spinLen:self.traintimeLen]
        )

        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)
        denominator = exp_output + exp_loss + exp_remember

        output_scale = exp_output / denominator
        loss_scale = exp_loss / denominator

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            if precipitation_scale is not None:
                precipitation = precipitation * precipitation_scale

            pet = x[0, b, 1].reshape(1, 1)
            storage_scaled = (storage - cmean) / cstd

            output_state = torch.addmm(
                output_bias,
                storage_scaled,
                self.weight_b1_yom,
            )
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_storage_state = torch.mm(
                storage_scaled,
                self.weight_b1_ylm,
            )
            loss_pet_state = torch.addmm(
                loss_bias,
                (pet - _PET_MEAN) / _PET_STD,
                self.weight_b2_ylm,
            )
            loss_fraction = loss_scale * torch.sigmoid(
                loss_storage_state + loss_pet_state
            )

            constrained_loss_fraction = _loss_constraint(
                loss_fraction,
                pet,
                storage,
                self.relu_l,
            )

            remember_fraction = (
                1.0 - output_fraction - constrained_loss_fraction
            )

            # Preserve historical PRE-update timing.
            output_flux = output_fraction * storage
            loss_flux = loss_fraction * storage
            constrained_loss_flux = constrained_loss_fraction * storage

            h_n[b, :] = output_flux[0]
            c_n[b, :] = storage[0]
            l_n[b, :] = loss_flux[0]
            lc_n[b, :] = constrained_loss_flux[0]
            bp_n[b, :] = 0.0

            gate_ib[b, :] = 0.0
            gate_oo[b, :] = output_fraction[0]
            gate_ol[b, :] = loss_fraction[0]
            gate_olc[b, :] = constrained_loss_fraction[0]
            gate_f[b, :] = remember_fraction[0]
            obs_std[b, :] = obsstd

            storage = remember_fraction * storage + precipitation

        return (
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_ib,
            gate_oo,
            gate_ol,
            gate_olc,
            gate_f,
            obs_std,
        )


class MCPBRNN_Generic_PETconstraint_LossMultiInput(_LossMultiInputBase):
    """Historical single-node MCP with storage + PET dependence in the loss gate."""

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        if hidden_size != 1:
            raise ValueError(
                "This historical Miscellaneous/SingleNode-MN case "
                "was implemented with hidden_size=1."
            )

        self._initialize_common_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self._reset_common_parameters()

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
        del epoch
        result = self._run_core(
            x,
            time_lag,
            y_obs,
            cmean,
            cstd,
        )

        h_n = result[0]
        obs_std = result[10]
        h_nout = torch.cat((h_n, obs_std), dim=1)

        return (
            h_n,
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
            result[7],
            result[8],
            result[9],
            h_nout,
            obs_std,
        )


class MCPBRNN_Generic_PETconstraint_LossMultiInput_DI(_LossMultiInputBase):
    """Historical DI variant with a learned positive precipitation-input weight."""

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        if hidden_size != 1:
            raise ValueError(
                "This historical Miscellaneous/SingleNode-DI case "
                "was implemented with hidden_size=1."
            )

        self._initialize_common_parameters()
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self._reset_common_parameters()
            self.wi.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
        del epoch

        result = self._run_core(
            x,
            time_lag,
            y_obs,
            cmean,
            cstd,
            precipitation_scale=torch.exp(self.wi),
        )

        h_n = result[0]

        return (
            h_n,
            h_n,
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
            result[7],
            result[8],
            result[9],
            result[10],
        )


class MCPBRNN_Generic_PETconstraint_LossMultiInput_DS(_LossMultiInputBase):
    """Historical DS variant with a learned positive output multiplier."""

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        if hidden_size != 1:
            raise ValueError(
                "This historical Miscellaneous/SingleNode-DS case "
                "was implemented with hidden_size=1."
            )

        self._initialize_common_parameters()
        self.wo = nn.Parameter(torch.empty(1, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self._reset_common_parameters()
            self.wo.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
        del epoch

        result = self._run_core(
            x,
            time_lag,
            y_obs,
            cmean,
            cstd,
        )

        h_n = result[0]

        # Preserve the historical DS implementation exactly:
        # Q = h_n @ exp(wo)
        hnout = torch.mm(
            h_n,
            torch.exp(self.wo),
        )

        return (
            hnout,
            h_n,
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
            result[7],
            result[8],
            result[9],
            result[10],
        )

class MCPBRNN_Generic_PETconstraint_LossMultiInput_MLB(_LossMultiInputBase):
    """Historical MLB variant with learned input/output scaling and output bias."""

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        if hidden_size != 1:
            raise ValueError(
                "This historical Miscellaneous/SingleNode-MLB case "
                "was implemented with hidden_size=1."
            )

        self._initialize_common_parameters()

        # Preserve original checkpoint parameter names and shapes.
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.wo = nn.Parameter(torch.empty(1, hidden_size))
        self.bias = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self._reset_common_parameters()
            self.wi.uniform_(0.0, 1.0)
            self.wo.uniform_(0.0, 1.0)
            self.bias.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
        del epoch

        result = self._run_core(
            x,
            time_lag,
            y_obs,
            cmean,
            cstd,
            precipitation_scale=torch.exp(self.wi),
        )

        h_n = result[0]

        # Preserve the historical MLB implementation:
        # Q = bias + h_n @ exp(wo)
        bias_expanded = self.bias.unsqueeze(0).expand(
            h_n.shape[0],
            1,
        )
        hnout = torch.addmm(
            bias_expanded,
            h_n,
            torch.exp(self.wo),
        )

        return (
            hnout,
            h_n,
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
            result[7],
            result[8],
            result[9],
            result[10],
        )


class MCN_SingleMultiML_IC(nn.Module):
    """
    Parallel MCP nodes without cross-node sharing in the state-dependent gates.

    Each node has its own scalar storage dependence for the output and loss
    gates. The node discharges are combined by a learned linear output layer.
    """

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
        self.initial_forget_bias = initial_forget_bias

        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        self.wo = nn.Parameter(torch.empty(hidden_size, 1))
        self.bias = nn.Parameter(torch.empty(1, 1))

        self.relu_l = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.bias.uniform_(0.0, 1.0)
            self.wo.uniform_(0.0, 1.0)

            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, cmean, cstd):
        x, batch_size = _validate_input(x, self.batch_first)

        if cstd == 0:
            raise ValueError("cstd must be non-zero.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        l_n = x.new_zeros(batch_size, hidden_size)
        lc_n = x.new_zeros(batch_size, hidden_size)
        bp_n = x.new_zeros(batch_size, hidden_size)

        gate_ib = x.new_zeros(batch_size, hidden_size)
        gate_oo = x.new_zeros(batch_size, hidden_size)
        gate_ol = x.new_zeros(batch_size, hidden_size)
        gate_olc = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)

        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)
        denominator = exp_output + exp_loss + exp_remember

        output_scale = exp_output / denominator
        loss_scale = exp_loss / denominator

        for b in range(batch_size):
            precipitation = x[0, b, 0].reshape(1, 1).expand(1, hidden_size)
            pet = x[0, b, 1].reshape(1, 1).expand(1, hidden_size)

            storage_scaled = (storage - cmean) / cstd

            output_state = (
                output_bias
                + storage_scaled * self.weight_b1_yom
            )
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_storage_state = storage_scaled * self.weight_b1_ylm
            loss_pet_state = (
                loss_bias
                + ((pet - _PET_MEAN) / _PET_STD) * self.weight_b2_ylm
            )
            loss_fraction = loss_scale * torch.sigmoid(
                loss_storage_state + loss_pet_state
            )

            constrained_loss_fraction = _loss_constraint(
                loss_fraction,
                pet,
                storage,
                self.relu_l,
            )

            remember_fraction = (
                1.0 - output_fraction - constrained_loss_fraction
            )

            # Historical PRE-update timing.
            output_flux = output_fraction * storage
            loss_flux = loss_fraction * storage
            constrained_loss_flux = constrained_loss_fraction * storage

            h_n[b, :] = output_flux[0]
            c_n[b, :] = storage[0]
            l_n[b, :] = loss_flux[0]
            lc_n[b, :] = constrained_loss_flux[0]
            bp_n[b, :] = 0.0

            gate_ib[b, :] = 0.0
            gate_oo[b, :] = output_fraction[0]
            gate_ol[b, :] = loss_fraction[0]
            gate_olc[b, :] = constrained_loss_fraction[0]
            gate_f[b, :] = remember_fraction[0]

            storage = remember_fraction * storage + precipitation

        total_q = torch.addmm(self.bias, h_n, self.wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_ib,
            gate_oo,
            gate_ol,
            gate_olc,
            gate_f,
        )


class MCN_SingleMultiML_Single_IC(nn.Module):
    """
    Single MCP nodes without cross-node sharing in the state-dependent gates.

    Each node has its own scalar storage dependence for the output and loss
    gates. The node discharges are combined by a learned linear output layer.
    """

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
        self.initial_forget_bias = initial_forget_bias

        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        self.wo = nn.Parameter(torch.empty(hidden_size, 1))
        self.bias = nn.Parameter(torch.empty(1))

        self.relu_l = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.bias.uniform_(0.0, 1.0)
            self.wo.uniform_(0.0, 1.0)

            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, cmean, cstd):
        x, batch_size = _validate_input(x, self.batch_first)

        if cstd == 0:
            raise ValueError("cstd must be non-zero.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        l_n = x.new_zeros(batch_size, hidden_size)
        lc_n = x.new_zeros(batch_size, hidden_size)
        bp_n = x.new_zeros(batch_size, hidden_size)

        gate_ib = x.new_zeros(batch_size, hidden_size)
        gate_oo = x.new_zeros(batch_size, hidden_size)
        gate_ol = x.new_zeros(batch_size, hidden_size)
        gate_olc = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)

        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)
        denominator = exp_output + exp_loss + exp_remember

        output_scale = exp_output / denominator
        loss_scale = exp_loss / denominator

        for b in range(batch_size):
            precipitation = x[0, b, 0].reshape(1, 1).expand(1, hidden_size)
            pet = x[0, b, 1].reshape(1, 1).expand(1, hidden_size)

            storage_scaled = (storage - cmean) / cstd

            output_state = (
                output_bias
                + storage_scaled * self.weight_b1_yom
            )
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_storage_state = storage_scaled * self.weight_b1_ylm
            loss_pet_state = (
                loss_bias
                + ((pet - _PET_MEAN) / _PET_STD) * self.weight_b2_ylm
            )
            loss_fraction = loss_scale * torch.sigmoid(
                loss_storage_state + loss_pet_state
            )

            constrained_loss_fraction = _loss_constraint(
                loss_fraction,
                pet,
                storage,
                self.relu_l,
            )

            remember_fraction = (
                1.0 - output_fraction - constrained_loss_fraction
            )

            # Historical PRE-update timing.
            output_flux = output_fraction * storage
            loss_flux = loss_fraction * storage
            constrained_loss_flux = constrained_loss_fraction * storage

            h_n[b, :] = output_flux[0]
            c_n[b, :] = storage[0]
            l_n[b, :] = loss_flux[0]
            lc_n[b, :] = constrained_loss_flux[0]
            bp_n[b, :] = 0.0

            gate_ib[b, :] = 0.0
            gate_oo[b, :] = output_fraction[0]
            gate_ol[b, :] = loss_fraction[0]
            gate_olc[b, :] = constrained_loss_fraction[0]
            gate_f[b, :] = remember_fraction[0]

            storage = remember_fraction * storage + precipitation

        total_q = torch.addmm(self.bias.view(1, 1), h_n, self.wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_ib,
            gate_oo,
            gate_ol,
            gate_olc,
            gate_f,
        )


class MCN_SingleMultiML_SharedLossOutput_IC(nn.Module):
    """
    Parallel MCP nodes with cross-node sharing in the state-dependent gates.

    The output and loss gates use full hidden_size x hidden_size matrices,
    allowing each node's gate to depend on the storage states of all nodes.
    The node discharges are combined by a learned linear output layer.
    """

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
        self.initial_forget_bias = initial_forget_bias

        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        self.wo = nn.Parameter(torch.empty(hidden_size, 1))
        self.bias = nn.Parameter(torch.empty(1, 1))

        self.relu_l = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.bias.uniform_(0.0, 1.0)
            self.wo.uniform_(0.0, 1.0)

            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, cmean, cstd):
        x, batch_size = _validate_input(x, self.batch_first)

        if cstd == 0:
            raise ValueError("cstd must be non-zero.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        h_n = x.new_zeros(batch_size, hidden_size)
        c_n = x.new_zeros(batch_size, hidden_size)
        l_n = x.new_zeros(batch_size, hidden_size)
        lc_n = x.new_zeros(batch_size, hidden_size)
        bp_n = x.new_zeros(batch_size, hidden_size)

        gate_ib = x.new_zeros(batch_size, hidden_size)
        gate_oo = x.new_zeros(batch_size, hidden_size)
        gate_ol = x.new_zeros(batch_size, hidden_size)
        gate_olc = x.new_zeros(batch_size, hidden_size)
        gate_f = x.new_zeros(batch_size, hidden_size)

        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)
        denominator = exp_output + exp_loss + exp_remember

        output_scale = exp_output / denominator
        loss_scale = exp_loss / denominator

        for b in range(batch_size):
            precipitation = x[0, b, 0].reshape(1, 1).expand(1, hidden_size)
            pet = x[0, b, 1].reshape(1, 1).expand(1, hidden_size)

            storage_scaled = (storage - cmean) / cstd

            output_state = torch.addmm(
                output_bias,
                storage_scaled,
                self.weight_b1_yom,
            )
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_storage_state = torch.mm(
                storage_scaled,
                self.weight_b1_ylm,
            )
            loss_pet_state = (
                loss_bias
                + ((pet - _PET_MEAN) / _PET_STD) * self.weight_b2_ylm
            )
            loss_fraction = loss_scale * torch.sigmoid(
                loss_storage_state + loss_pet_state
            )

            constrained_loss_fraction = _loss_constraint(
                loss_fraction,
                pet,
                storage,
                self.relu_l,
            )

            remember_fraction = (
                1.0 - output_fraction - constrained_loss_fraction
            )

            # Historical PRE-update timing.
            output_flux = output_fraction * storage
            loss_flux = loss_fraction * storage
            constrained_loss_flux = constrained_loss_fraction * storage

            h_n[b, :] = output_flux[0]
            c_n[b, :] = storage[0]
            l_n[b, :] = loss_flux[0]
            lc_n[b, :] = constrained_loss_flux[0]
            bp_n[b, :] = 0.0

            gate_ib[b, :] = 0.0
            gate_oo[b, :] = output_fraction[0]
            gate_ol[b, :] = loss_fraction[0]
            gate_olc[b, :] = constrained_loss_fraction[0]
            gate_f[b, :] = remember_fraction[0]

            storage = remember_fraction * storage + precipitation

        total_q = torch.addmm(self.bias, h_n, self.wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_ib,
            gate_oo,
            gate_ol,
            gate_olc,
            gate_f,
        )