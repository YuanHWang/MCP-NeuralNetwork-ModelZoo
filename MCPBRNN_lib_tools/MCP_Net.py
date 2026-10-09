import torch
import torch.nn as nn
from torch import Tensor

class MCPNet_DI_None_SingleLayer(nn.Module):
    """Single-layer Distributed-Input MCP network without information sharing."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("MCPNet_DI_None_SingleLayer requires seq_len == 1.")
        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DI: positive precipitation fractions that sum to one.
        wi = torch.exp(self.wi)
        wi = wi / wi.sum()

        # MCP output/loss/remember allocation.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)
        denom = er_o + er_l + er_r
        scale_o, scale_l = er_o / denom, er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)
            u = p * wi

            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std) * self.weight_b1_yom
            )

            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std) * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980) * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)
            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j] - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update diagnostics.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n are kept explicitly for MCP interpretability.
            c = remember * c + u

        total_q = h_n.sum(dim=1, keepdim=True)

        return (
            total_q, h_n, c_n, l_n, lc_n, bp_n,
            gate_i, gate_o, gate_l, gate_lc, gate_r
        )

class MCPNet_DI_SAL_SingleLayer(nn.Module):
    """Single-layer DI MCP network with sharing-augmented loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # SAL: every loss gate receives information from all states.
        self.weight_b1_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("MCPNet_DI_SAL_SingleLayer requires seq_len == 1.")
        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DI precipitation allocation.
        wi = torch.exp(self.wi)
        wi = wi / wi.sum()

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)
        denom = er_o + er_l + er_r
        scale_o, scale_l = er_o / denom, er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)
            u = p * wi

            # Output gate: local-state information only.
            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std) * self.weight_b1_yom
            )

            # Loss gate: shared information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm((c - c_mean) / c_std, self.weight_b1_ylm)
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980) * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)
            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j] - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n are retained explicitly for interpretability.
            c = remember * c + u

        total_q = h_n.sum(dim=1, keepdim=True)

        return (
            total_q, h_n, c_n, l_n, lc_n, bp_n,
            gate_i, gate_o, gate_l, gate_lc, gate_r
        )

class MCPNet_DI_SALO_SingleLayer(nn.Module):
    """Single-layer DI MCP network with sharing at output and loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SALO: output and loss gates both receive all node states.
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("MCPNet_DI_SALO_SingleLayer requires seq_len == 1.")
        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DI: positive precipitation fractions that sum to one.
        wi = torch.exp(self.wi)
        wi = wi / wi.sum()

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)
        denom = er_o + er_l + er_r
        scale_o, scale_l = er_o / denom, er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)
            u = p * wi

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm((c - c_mean) / c_std, self.weight_b1_yom)
                + self.bias_b0_yom.unsqueeze(0)
            )

            # SAL: loss gate receives information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm((c - c_mean) / c_std, self.weight_b1_ylm)
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980) * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)
            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j] - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        total_q = h_n.sum(dim=1, keepdim=True)

        return (
            total_q, h_n, c_n, l_n, lc_n, bp_n,
            gate_i, gate_o, gate_l, gate_lc, gate_r
        )

class MCPNet_DI_SAO_SingleLayer(nn.Module):
    """Single-layer DI MCP network with sharing at the output gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SAO: every output gate receives information from all states.
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("MCPNet_DI_SAO_SingleLayer requires seq_len == 1.")
        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DI: positive precipitation fractions that sum to one.
        wi = torch.exp(self.wi)
        wi = wi / wi.sum()

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)
        denom = er_o + er_l + er_r
        scale_o, scale_l = er_o / denom, er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)
            u = p * wi

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm((c - c_mean) / c_std, self.weight_b1_yom)
                + self.bias_b0_yom.unsqueeze(0)
            )

            # Loss gate uses only its own node state.
            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std) * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980) * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)
            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j] - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        total_q = h_n.sum(dim=1, keepdim=True)

        return (
            total_q, h_n, c_n, l_n, lc_n, bp_n,
            gate_i, gate_o, gate_l, gate_lc, gate_r
        )

class MCPNet_DIR_None_SingleLayer(nn.Module):
    """Single-layer Distributed-Input-Relaxed MCP network without information sharing."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError("MCPNet_DIR_None_SingleLayer requires seq_len == 1.")
        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DIR: positive input multipliers, not constrained to sum to one.
        wi = torch.exp(self.wi)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)
        denom = er_o + er_l + er_r
        scale_o, scale_l = er_o / denom, er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # Unlike DI, these weights are not normalized.
            u = p * wi

            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std) * self.weight_b1_yom
            )

            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std) * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980) * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)
            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j] - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        total_q = h_n.sum(dim=1, keepdim=True)

        return (
            total_q, h_n, c_n, l_n, lc_n, bp_n,
            gate_i, gate_o, gate_l, gate_lc, gate_r
        )


class MCPNet_DIR_SAL_SingleLayer(nn.Module):
    """Single-layer DIR MCP network with sharing at the loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # SAL: every loss gate receives information from all node states.
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        if seq_len != 1:
            raise ValueError(
                "MCPNet_DIR_SAL_SingleLayer requires seq_len == 1."
            )
        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DIR: positive input multipliers, not constrained to sum to one.
        wi = torch.exp(self.wi)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)
        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # DIR input allocation.
            u = p * wi

            # Output gate: local-state information only.
            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std) * self.weight_b1_yom
            )

            # SAL: loss gate receives information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_ylm,
                )
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        total_q = h_n.sum(dim=1, keepdim=True)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DIR_SALO_SingleLayer(nn.Module):
    """Single-layer DIR MCP network with sharing at output and loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SALO: both output and loss gates share all node states.
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wi = nn.Parameter(torch.empty(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DIR_SALO_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DIR: positive input multipliers, not constrained to sum to one.
        wi = torch.exp(self.wi)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # DIR input allocation.
            u = p * wi

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_yom,
                )
                + self.bias_b0_yom.unsqueeze(0)
            )

            # SAL: loss gate receives information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_ylm,
                )
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        total_q = h_n.sum(dim=1, keepdim=True)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DIR_SAO_SingleLayer(nn.Module):
    """Single-layer DIR MCP network with sharing at the output gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SAO: every output gate receives information from all states.
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.wi = nn.Parameter(torch.empty(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DIR_SAO_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DIR: positive input multipliers, not constrained to sum to one.
        wi = torch.exp(self.wi)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # DIR input allocation.
            u = p * wi

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_yom,
                )
                + self.bias_b0_yom.unsqueeze(0)
            )

            # Loss gate uses only its own node state.
            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std)
                * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        total_q = h_n.sum(dim=1, keepdim=True)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DS_None_SingleLayer(nn.Module):
    """Single-layer Distributed-State MCP network without information sharing."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # DS: positive output weights constrained to sum to one.
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DS_None_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DS: every node receives the full precipitation input.
        # Network-level output weights are positive and sum to one.
        wo = torch.exp(self.wo)
        wo = wo / wo.sum()

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # Full precipitation enters every node.
            u = p

            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std)
                * self.weight_b1_yom
            )

            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std)
                * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # DS network-level discharge:
        # positive output weights constrained to sum to one.
        total_q = torch.mm(h_n, wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DS_SAL_SingleLayer(nn.Module):
    """Single-layer DS MCP network with sharing at the loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # SAL: every loss gate receives information from all node states.
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # DS: positive output weights constrained to sum to one.
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DS_SAL_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DS: positive output weights that sum to one.
        wo = torch.exp(self.wo)
        wo = wo / wo.sum()

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # DS: every node receives the full precipitation input.
            u = p

            # Output gate: local-state information only.
            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std)
                * self.weight_b1_yom
            )

            # SAL: loss gate receives information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_ylm,
                )
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # DS network-level discharge.
        total_q = torch.mm(h_n, wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )


class MCPNet_DS_SALO_SingleLayer(nn.Module):
    """Single-layer DS MCP network with sharing at output and loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SALO: output and loss gates both receive all node states.
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        self.weight_b1_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # DS: positive output weights constrained to sum to one.
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DS_SALO_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DS: positive output weights that sum to one.
        wo = torch.exp(self.wo)
        wo = wo / wo.sum()

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # DS: every node receives the full precipitation input.
            u = p

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_yom,
                )
                + self.bias_b0_yom.unsqueeze(0)
            )

            # SAL: loss gate receives information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_ylm,
                )
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # DS network-level discharge.
        total_q = torch.mm(h_n, wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DS_SAO_SingleLayer(nn.Module):
    """Single-layer DS MCP network with sharing at the output gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SAO: output gate receives information from all node states.
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # Loss gate remains local.
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # DS: positive output weights constrained to sum to one.
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DS_SAO_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DS: positive network-level output weights that sum to one.
        wo = torch.exp(self.wo)
        wo = wo / wo.sum()

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # DS: every node receives the full precipitation input.
            u = p

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_yom,
                )
                + self.bias_b0_yom.unsqueeze(0)
            )

            # Loss gate uses only its own node state.
            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std)
                * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # DS network-level discharge.
        total_q = torch.mm(h_n, wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DSR_None_SingleLayer(nn.Module):
    """Single-layer DSR MCP network without information sharing."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # DSR: positive output multipliers without sum-to-one constraint.
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DSR_None_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DSR: positive output multipliers, not normalized.
        wo = torch.exp(self.wo)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # Every node receives the full precipitation input.
            u = p

            # Output gate: local-state information only.
            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std)
                * self.weight_b1_yom
            )

            # Loss gate: local-state information only.
            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std)
                * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # DSR: positive but unconstrained network-level output aggregation.
        total_q = torch.mm(h_n, wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DSR_SAL_SingleLayer(nn.Module):
    """Single-layer DSR MCP network with sharing at the loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # SAL: every loss gate receives information from all node states.
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # DSR: positive output multipliers without sum-to-one constraint.
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DSR_SAL_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DSR: positive output multipliers, not normalized.
        wo = torch.exp(self.wo)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # Every node receives the full precipitation input.
            u = p

            # Output gate remains local.
            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std)
                * self.weight_b1_yom
            )

            # SAL: loss gate receives information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_ylm,
                )
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # DSR: positive but unconstrained output aggregation.
        total_q = torch.mm(h_n, wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DSR_SALO_SingleLayer(nn.Module):
    """Single-layer DSR MCP network with sharing at output and loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SAO: output gate receives all node states.
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # SAL: loss gate receives all node states.
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # DSR: positive output multipliers without sum-to-one constraint.
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DSR_SALO_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DSR: positive output multipliers, not normalized.
        wo = torch.exp(self.wo)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # Every node receives the full precipitation input.
            u = p

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_yom,
                )
                + self.bias_b0_yom.unsqueeze(0)
            )

            # SAL: loss gate receives information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_ylm,
                )
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # DSR: positive but unconstrained output aggregation.
        total_q = torch.mm(h_n, wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DSR_SAO_SingleLayer(nn.Module):
    """Single-layer DSR MCP network with sharing at the output gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SAO: every output gate receives information from all states.
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # Loss gate remains local.
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # DSR: positive output multipliers without sum-to-one constraint.
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DSR_SAO_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DSR: positive output multipliers, not normalized.
        wo = torch.exp(self.wo)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # Every node receives the full precipitation input.
            u = p

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_yom,
                )
                + self.bias_b0_yom.unsqueeze(0)
            )

            # Loss gate uses only its own node state.
            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std)
                * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # DSR: positive but unconstrained output aggregation.
        total_q = torch.mm(h_n, wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_DSR_SAO_SingleLayer(nn.Module):
    """Single-layer DSR MCP network with sharing at the output gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SAO: every output gate receives information from all states.
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # Loss gate remains local.
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # DSR: positive output multipliers without sum-to-one constraint.
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_DSR_SAO_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # DSR: positive output multipliers, not normalized.
        wo = torch.exp(self.wo)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # Every node receives the full precipitation input.
            u = p

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_yom,
                )
                + self.bias_b0_yom.unsqueeze(0)
            )

            # Loss gate uses only its own node state.
            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std)
                * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # DSR: positive but unconstrained output aggregation.
        total_q = torch.mm(h_n, wo)

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_MLB_None_SingleLayer(nn.Module):
    """Single-layer MLB MCP network without information sharing."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # MLB network-level input/output mappings.
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))
        self.bias = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_MLB_None_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # MLB input mapping: positive but not normalized.
        wi = torch.exp(self.wi)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # MLB input transformation.
            u = p * wi

            # Output gate: local-state information only.
            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std)
                * self.weight_b1_yom
            )

            # Loss gate: local-state information only.
            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std)
                * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # MLB network-level output mapping:
        # unconstrained linear weights plus additive bias.
        total_q = torch.addmm(
            self.bias.view(1, 1).expand(batch_size, 1),
            h_n,
            self.wo,
        )

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_MLB_SAL_SingleLayer(nn.Module):
    """Single-layer MLB MCP network with sharing at the loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # SAL: every loss gate receives information from all states.
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # MLB network-level input/output mappings.
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))
        self.bias = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_MLB_SAL_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # MLB input mapping: positive but not normalized.
        wi = torch.exp(self.wi)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # MLB input transformation.
            u = p * wi

            # Output gate remains local.
            oo = scale_o * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + ((c - c_mean) / c_std)
                * self.weight_b1_yom
            )

            # SAL: loss gate receives information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_ylm,
                )
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # MLB network-level output:
        # general linear weights plus additive bias.
        total_q = torch.addmm(
            self.bias.view(1, 1).expand(batch_size, 1),
            h_n,
            self.wo,
        )

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_MLB_SALO_SingleLayer(nn.Module):
    """Single-layer MLB MCP network with sharing at output and loss gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SAO: output gate receives all node states.
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # SAL: loss gate receives all node states.
        self.weight_b1_ylm = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # MLB network-level input/output mappings.
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))
        self.bias = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_MLB_SALO_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # MLB input mapping: positive but not normalized.
        wi = torch.exp(self.wi)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # MLB input transformation.
            u = p * wi

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_yom,
                )
                + self.bias_b0_yom.unsqueeze(0)
            )

            # SAL: loss gate receives information from all states.
            ol = scale_l * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_ylm,
                )
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # MLB network-level output:
        # general linear weights plus additive bias.
        total_q = torch.addmm(
            self.bias.view(1, 1).expand(batch_size, 1),
            h_n,
            self.wo,
        )

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )

class MCPNet_MLB_SAO_SingleLayer(nn.Module):
    """Single-layer MLB MCP network with sharing at the output gates."""

    def __init__(self, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Legacy parameter names/shapes retained for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(1, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))

        # SAO: every output gate receives information from all states.
        self.weight_b1_yom = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))

        # Loss gate remains local.
        self.weight_b1_ylm = nn.Parameter(torch.empty(1, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(1, hidden_size))

        # MLB network-level input/output mappings.
        self.wi = nn.Parameter(torch.empty(1, hidden_size))
        self.wo = nn.Parameter(torch.empty(hidden_size, 1))
        self.bias = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand_like(p))

    def forward(self, x, c_mean, c_std):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError(
                "MCPNet_MLB_SAO_SingleLayer requires seq_len == 1."
            )

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        n = self.hidden_size
        c = x.new_zeros(1, n)

        h_n = x.new_zeros(batch_size, n)
        c_n = x.new_zeros(batch_size, n)
        l_n = x.new_zeros(batch_size, n)
        lc_n = x.new_zeros(batch_size, n)
        bp_n = x.new_zeros(batch_size, n)

        gate_i = x.new_zeros(batch_size, n)
        gate_o = x.new_zeros(batch_size, n)
        gate_l = x.new_zeros(batch_size, n)
        gate_lc = x.new_zeros(batch_size, n)
        gate_r = x.new_zeros(batch_size, n)

        # MLB input mapping: positive but not normalized.
        wi = torch.exp(self.wi)

        # MCP mass-allocation scales.
        er_o = torch.exp(self.weight_r_yom)
        er_l = torch.exp(self.weight_r_ylm)
        er_r = torch.exp(self.weight_r_yfm)

        denom = er_o + er_l + er_r
        scale_o = er_o / denom
        scale_l = er_l / denom

        for b in range(batch_size):
            p = x[0, b, 0].view(1, 1).expand(1, n)
            pet = x[0, b, 1].view(1, 1).expand(1, n)

            # MLB input transformation.
            u = p * wi

            # SAO: output gate receives information from all states.
            oo = scale_o * torch.sigmoid(
                torch.mm(
                    (c - c_mean) / c_std,
                    self.weight_b1_yom,
                )
                + self.bias_b0_yom.unsqueeze(0)
            )

            # Loss gate uses only its own node state.
            ol = scale_l * torch.sigmoid(
                ((c - c_mean) / c_std)
                * self.weight_b1_ylm
                + self.bias_b0_ylm.unsqueeze(0)
                + ((pet - 2.9086) / 1.8980)
                * self.weight_b2_ylm
            )

            olc = x.new_zeros(1, n)

            for j in range(n):
                if c[0, j] > 0:
                    olc[0, j] = ol[0, j] - torch.relu(
                        ol[0, j]
                        - pet[0, j] / c[0, j]
                    )
                else:
                    olc[0, j] = ol[0, j]

            remember = 1.0 - oo - olc

            # PRE-update states, fluxes, and gates.
            h_n[b] = oo * c
            c_n[b] = c
            l_n[b] = ol * c
            lc_n[b] = olc * c

            gate_o[b] = oo
            gate_l[b] = ol
            gate_lc[b] = olc
            gate_r[b] = remember

            # gate_i and bp_n retained explicitly for interpretability.
            c = remember * c + u

        # MLB network-level output:
        # general linear weights plus additive bias.
        total_q = torch.addmm(
            self.bias.view(1, 1).expand(batch_size, 1),
            h_n,
            self.wo,
        )

        return (
            total_q,
            h_n,
            c_n,
            l_n,
            lc_n,
            bp_n,
            gate_i,
            gate_o,
            gate_l,
            gate_lc,
            gate_r,
        )