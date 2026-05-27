"""
TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting
Adapted from: https://github.com/Atik-Ahamed/TimeMachine

Modified for FTMamba's TSL framework:
- Uses pure PyTorch MambaBlock instead of mamba_ssm dependency
- Uses instance normalization instead of RevIN (consistent with FTMamba)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat, einsum

try:
    from mamba_ssm import Mamba as MambaSSM
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


class MambaBlock(nn.Module):
    """Pure PyTorch Mamba block (from FTMamba implementation)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), "n -> d n", d=self.d_inner).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        y = self._ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def _ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        return self._selective_scan(x, delta, A, B, C, D)

    def _selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n").clamp(max=10.0))
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            x = x.clamp(min=-1e4, max=1e4)
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y


class Model(nn.Module):
    """
    TimeMachine: 4 parallel Mamba blocks at different scales.
    Adapted to work with TSL run.py framework.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        # Config parameters with defaults
        self.n1 = getattr(configs, 'tm_n1', 96)
        self.n2 = getattr(configs, 'tm_n2', 64)
        self.d_state = getattr(configs, 'd_ff', 16)
        self.d_conv = getattr(configs, 'd_conv', 4)
        self.expand = getattr(configs, 'expand', 2)
        self.ch_ind = getattr(configs, 'tm_ch_ind', 1)
        self.use_residual = getattr(configs, 'tm_residual', 1)

        self.lin1 = nn.Linear(configs.seq_len, self.n1)
        self.dropout1 = nn.Dropout(configs.dropout)

        self.lin2 = nn.Linear(self.n1, self.n2)
        self.dropout2 = nn.Dropout(configs.dropout)

        if self.ch_ind == 1:
            self.d_model_param1 = 1
            self.d_model_param2 = 1
        else:
            self.d_model_param1 = self.n2
            self.d_model_param2 = self.n1

        self.mamba1 = MambaBlock(d_model=self.d_model_param1, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.mamba2 = MambaBlock(d_model=self.n2, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.mamba3 = MambaBlock(d_model=self.n1, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.mamba4 = MambaBlock(d_model=self.d_model_param2, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)

        self.lin3 = nn.Linear(self.n2, self.n1)
        self.lin4 = nn.Linear(2 * self.n1, configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x = x_enc

        # Instance normalization (same as FTMamba)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = torch.permute(x, (0, 2, 1))
        if self.ch_ind == 1:
            x = torch.reshape(x, (x.shape[0] * x.shape[1], 1, x.shape[2]))

        x = self.lin1(x)
        x_res1 = x
        x = self.dropout1(x)
        x3 = self.mamba3(x)

        if self.ch_ind == 1:
            x4 = torch.permute(x, (0, 2, 1))
        else:
            x4 = x
        x4 = self.mamba4(x4)
        if self.ch_ind == 1:
            x4 = torch.permute(x4, (0, 2, 1))

        x4 = x4 + x3

        x = self.lin2(x)
        x_res2 = x
        x = self.dropout2(x)

        if self.ch_ind == 1:
            x1 = torch.permute(x, (0, 2, 1))
        else:
            x1 = x
        x1 = self.mamba1(x1)
        if self.ch_ind == 1:
            x1 = torch.permute(x1, (0, 2, 1))

        x2 = self.mamba2(x)

        if self.use_residual == 1:
            x = x1 + x_res2 + x2
        else:
            x = x1 + x2

        x = self.lin3(x)
        if self.use_residual == 1:
            x = x + x_res1

        x = torch.cat([x, x4], dim=2)
        x = self.lin4(x)

        if self.ch_ind == 1:
            x = torch.reshape(x, (-1, self.configs.enc_in, self.configs.pred_len))

        x = torch.permute(x, (0, 2, 1))

        # De-normalize
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))

        return x
