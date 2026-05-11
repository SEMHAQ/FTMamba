"""
S-Mamba: Mamba as a Multivariate Time Series Forecaster
Paper: https://arxiv.org/abs/2403.11144

Applies Mamba across the variate dimension to capture inter-variate dependencies,
in contrast to standard Mamba which operates on the time dimension.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.d_conv = configs.d_conv
        self.expand = configs.expand
        self.e_layers = configs.e_layers

        self.embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Variate-level Mamba blocks
        self.variate_layers = nn.ModuleList([
            VariateMambaBlock(configs) for _ in range(configs.e_layers)
        ])

        # Prediction head
        self.norm = nn.LayerNorm(configs.d_model)
        self.out_layer = nn.Linear(configs.d_model, configs.pred_len, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        # Instance normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        # Embedding: [B, L, C] -> [B, L, D]
        x = self.embedding(x_enc, x_mark_enc)

        # Apply variate-level Mamba blocks
        for layer in self.variate_layers:
            x = layer(x)

        x = self.norm(x)

        # Prediction head: [B, L, D] -> [B, pred_len, C]
        # Transpose to apply linear across time dimension
        x_out = self.out_layer(x.transpose(1, 2)).transpose(1, 2)

        # De-normalize
        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]


class VariateMambaBlock(nn.Module):
    """Applies Mamba across variates at each time step."""

    def __init__(self, configs):
        super(VariateMambaBlock, self).__init__()
        d_model = configs.d_model
        d_inner = d_model * configs.expand
        dt_rank = math.ceil(d_model / 16)
        d_conv = configs.d_conv
        d_state = configs.d_ff

        self.norm = nn.LayerNorm(d_model)
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        # Mamba components
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            bias=True, kernel_size=d_conv, padding=d_conv - 1,
            groups=d_inner,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: [B, L, D]
        Apply Mamba across the variate-like dimension.
        Here we treat the sequence dimension as the scan dimension.
        """
        residual = x
        x = self.norm(x)

        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        y = self._ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output + residual

    def _ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        y = self._selective_scan(x, delta, A, B, C, D)
        return y

    def _selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(torch.einsum("b l d, d n -> b l d n", delta, A))
        deltaB_u = torch.einsum(
            "b l d, b l n, b l d -> b l d n", delta, B, u
        )

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum("b d n, b n -> b d", x, C[:, i, :])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y
