"""Generate publication-quality figures for FTMamba paper.

Outputs:
  - fig_main_results.pdf: MSE comparison across 3 datasets (3 subplots)
  - fig_ablation.pdf: Ablation study bar chart
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
horizons = [96, 192, 336, 720]
models = ['FTMamba', 'PatchTST', 'iTransformer', 'DLinear', 'TimesNet', 'Transformer']
colors = ['#2171B5', '#6BAED6', '#BDD7E7', '#FDD49E', '#FDBB84', '#E34A33']
hatches = ['', '', '', '', '', '']

etth1 = {
    'FTMamba':     [0.3776, 0.4296, 0.4931, 0.4668],
    'PatchTST':    [0.3827, 0.4385, 0.4857, 0.4878],
    'iTransformer': [0.3935, 0.4521, 0.4941, 0.5118],
    'DLinear':     [0.4108, 0.4579, 0.4972, 0.5231],
    'TimesNet':    [0.4333, 0.5067, 0.5507, 0.7140],
    'Transformer': [0.8445, 0.8028, 1.0943, 1.0878],
}

etth2 = {
    'FTMamba':     [0.2904, 0.3784, 0.4118, 0.4434],
    'PatchTST':    [0.3016, 0.3757, 0.4211, 0.4424],
    'iTransformer': [0.3007, 0.3938, 0.4350, 0.4317],
    'DLinear':     [0.3595, 0.4917, 0.5985, 0.8597],
    'TimesNet':    [0.3444, 0.4327, 0.4729, 0.4637],
    'Transformer': [1.9057, 3.6999, 3.2220, 3.4519],
}

ettm1 = {
    'FTMamba':     [0.3439, 0.3757, 0.4099, 0.4697],
    'PatchTST':    [0.3368, 0.3683, 0.4062, 0.4648],
    'iTransformer': [0.3377, 0.3873, 0.4283, 0.5142],
    'DLinear':     [0.3480, 0.3845, 0.4149, 0.4733],
    'TimesNet':    [0.3881, 0.4212, 0.4650, 0.5273],
    'Transformer': [0.5832, 0.5914, 1.0038, 1.1324],
}

# ── Ablation data ─────────────────────────────────────────────────────
ablation_configs = ['Full', 'w/o Freq', 'w/o Gate']
ablation_mse = [0.3826, 0.3859, 0.3979]
ablation_mae = [0.4042, 0.4114, 0.4149]

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def plot_bars(ax, data, show_ylabel=False):
    """Draw grouped bars on a single axes."""
    n_h = len(horizons)
    x = np.arange(n_h)
    width = 0.13
    offsets = np.arange(len(models)) - (len(models) - 1) / 2
    for i, model in enumerate(models):
        vals = data[model]
        ax.bar(x + offsets[i] * width, vals, width,
               label=model if show_ylabel else None,
               color=colors[i], edgecolor='white',
               linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def draw_break(ax_bottom, ax_top):
    """Draw diagonal break marks between two stacked axes."""
    d = 0.015
    kwargs = dict(transform=ax_bottom.transAxes, color='k', clip_on=False,
                  linewidth=0.8)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_top.transAxes)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)


# ── Figure 1: Main results (3 columns, broken axis for ETTh2) ───────
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(7.2, 3.0), layout='constrained')
gs = gridspec.GridSpec(1, 3, wspace=0.45, figure=fig)

# (a) ETTh1 — single axes
ax_a = fig.add_subplot(gs[0, 0])
plot_bars(ax_a, etth1, show_ylabel=True)
ax_a.set_xlabel('Prediction Horizon')
ax_a.set_ylabel('MSE')
ax_a.set_title('(a) ETTh1', fontweight='bold', pad=6)
ax_a.set_ylim(0.35, 1.15)

# (b) ETTh2 — broken axis: bottom 0-1.0, top 1.5-4.0
gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1],
                                         hspace=0.08, height_ratios=[1, 1])
ax_b_top = fig.add_subplot(gs_b[0])
ax_b_bot = fig.add_subplot(gs_b[1], sharex=ax_b_top)

plot_bars(ax_b_bot, etth2)
plot_bars(ax_b_top, etth2)

ax_b_bot.set_ylim(0, 1.0)
ax_b_top.set_ylim(1.5, 4.0)

ax_b_bot.set_xlabel('Prediction Horizon')
ax_b_bot.set_ylabel('MSE')
ax_b_top.set_title('(b) ETTh2', fontweight='bold', pad=6)

# Hide x-axis on top subplot
plt.setp(ax_b_top.get_xticklabels(), visible=False)

draw_break(ax_b_bot, ax_b_top)

# (c) ETTm1 — single axes
ax_c = fig.add_subplot(gs[0, 2])
plot_bars(ax_c, ettm1)
ax_c.set_xlabel('Prediction Horizon')
ax_c.set_title('(c) ETTm1', fontweight='bold', pad=6)
ax_c.set_ylim(0.3, 1.2)

# Shared legend below the figure
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='white',
                           linewidth=0.5) for c in colors]
fig.legend(handles, models, loc='lower center', ncol=6, fontsize=7.5,
           framealpha=0.9, edgecolor='#cccccc',
           bbox_to_anchor=(0.5, -0.02))

fig.savefig('fig_main_results.pdf', bbox_inches='tight')
fig.savefig('fig_main_results.png', dpi=300)
print('Saved fig_main_results.pdf and .png')


# ── Figure 2: Ablation ───────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(3.5, 2.8))

x2 = np.arange(len(ablation_configs))
width2 = 0.35

bars_mse = ax2.bar(x2 - width2/2, ablation_mse, width2, label='MSE',
                    color='#2171B5', edgecolor='white', linewidth=0.5, zorder=3)
bars_mae = ax2.bar(x2 + width2/2, ablation_mae, width2, label='MAE',
                    color='#FDBB84', edgecolor='white', linewidth=0.5, zorder=3)

# Annotate MSE values
for bar, val in zip(bars_mse, ablation_mse):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars_mae, ablation_mae):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=7)

ax2.set_xticks(x2)
ax2.set_xticklabels(ablation_configs, fontsize=9)
ax2.set_ylabel('Metric Value')
ax2.set_title('Ablation on ETTh1 (horizon=96)', fontweight='bold', pad=6)
ax2.legend(framealpha=0.9, edgecolor='#cccccc')
ax2.grid(axis='y', alpha=0.3, zorder=0)
ax2.set_axisbelow(True)
ax2.set_ylim(0.37, 0.425)

fig2.tight_layout()
fig2.savefig('fig_ablation.pdf')
fig2.savefig('fig_ablation.png', dpi=300)
print('Saved fig_ablation.pdf and .png')
