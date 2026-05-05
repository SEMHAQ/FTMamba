"""
FTMamba: Frequency-aware Temporal Mamba for Long-term Time Series Forecasting

Key innovation: Dual-branch architecture combining Mamba (temporal) with
learnable frequency decomposition, fused via a gated mechanism.

- Temporal branch: Mamba blocks with linear complexity O(L)
- Frequency branch: FFT + learnable frequency filter + iFFT
- Gated fusion: learnable gate to adaptively combine both branches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat, einsum
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MambaBlock(nn.Module):
    """Pure PyTorch Mamba block (no mamba_ssm dependency)."""

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

        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y


class FrequencyBranch(nn.Module):
    """
    Frequency-domain feature extraction branch.
    Applies FFT, learnable frequency filtering, and iFFT.
    """

    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # Learnable frequency filter (complex-valued via real/imag parts)
        self.freq_filter_real = nn.Parameter(torch.ones(1, 1, seq_len // 2 + 1))
        self.freq_filter_imag = nn.Parameter(torch.zeros(1, 1, seq_len // 2 + 1))
        # Projection after frequency processing
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B * n_vars, patch_num, d_model]
        """
        # Apply FFT along the sequence (patch) dimension
        x_fft = torch.fft.rfft(x, dim=1)  # [B*nv, patch_num//2+1, d_model]

        # Apply learnable frequency filter
        freq_filter = torch.complex(self.freq_filter_real, self.freq_filter_imag)
        # Broadcast filter across d_model dimension
        x_fft = x_fft * freq_filter.transpose(1, 2)  # broadcast over d_model

        # iFFT back to time domain
        x_freq = torch.fft.irfft(x_fft, n=x.shape[1], dim=1)  # [B*nv, patch_num, d_model]

        # Projection and norm
        x_freq = self.proj(x_freq)
        x_freq = self.norm(x_freq)
        return x_freq


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism to combine temporal and frequency features.
    gate = sigmoid(W_g * [h_temp; h_freq])
    output = gate * h_temp + (1 - gate) * h_freq
    """

    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, h_temp, h_freq):
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_temp, h_freq], dim=-1)))
        return gate * h_temp + (1 - gate) * h_freq


class FTMambaLayer(nn.Module):
    """Single FTMamba layer: Mamba + Frequency + Gated Fusion."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, seq_len=96):
        super().__init__()
        self.temporal_branch = MambaBlock(d_model, d_state, d_conv, expand)
        self.frequency_branch = FrequencyBranch(d_model, seq_len)
        self.fusion = GatedFusion(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B * n_vars, patch_num, d_model]
        """
        h_temp = self.temporal_branch(x)
        h_freq = self.frequency_branch(x)
        h_fused = self.fusion(h_temp, h_freq)
        return self.norm(h_fused + x)  # residual connection


class Model(nn.Module):
    """
    FTMamba: Frequency-aware Temporal Mamba for Long-term Time Series Forecasting.

    Combines Mamba's linear-complexity temporal modeling with frequency-domain
    feature extraction via gated fusion, built on a patch-based architecture.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )

        # Compute patch_num for frequency branch
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)

        # Stacked FTMamba layers
        self.layers = nn.ModuleList([
            FTMambaLayer(
                d_model=configs.d_model,
                d_state=configs.d_ff,
                d_conv=configs.d_conv,
                expand=configs.expand,
                seq_len=self.patch_num,
            )
            for _ in range(configs.e_layers)
        ])

        # Prediction head
        self.head_nf = configs.d_model * self.patch_num
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = FlattenHead(
                configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout
            )
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.head = FlattenHead(
                configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout
            )
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patch embedding: [B, seq_len, n_vars] -> [B*n_vars, patch_num, d_model]
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # FTMamba layers
        for layer in self.layers:
            enc_out = layer(enc_out)

        # Reshape: [B*n_vars, patch_num, d_model] -> [B, n_vars, d_model, patch_num]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Prediction head
        dec_out = self.head(enc_out)  # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        for layer in self.layers:
            enc_out = layer(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        for layer in self.layers:
            enc_out = layer(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        for layer in self.layers:
            enc_out = layer(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
