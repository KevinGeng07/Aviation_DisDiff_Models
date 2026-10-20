import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# many sampled futures per window (e.g. sample_futures.py --n_samples 1000)
full_futures = pd.read_csv('../futures_heatmap.csv')

models = [
    '6_6_128_SWiDiT',
    '6_4_384_SWiDiT',
    '6_4_256_SWiDiT',
    '6_4_128_SWiDiT',
    '6_2_384_SWiDiT',
    '6_2_256_SWiDiT',
    '6_2_128_SWiDiT'
]

# custom panel titles; any model left out falls back to its name
titles = {
    '6_6_128_SWiDiT': 'SWiDiT_6_128',
    '6_4_384_SWiDiT': 'SWiDiT_4_384',
    '6_4_256_SWiDiT': 'SWiDiT_4_256',
    '6_4_128_SWiDiT': 'SWiDiT_4_128',
    '6_2_384_SWiDiT': 'SWiDiT_2_384',
    '6_2_256_SWiDiT': 'SWiDiT_2_256',
    '6_2_128_SWiDiT': 'SWiDiT_2_128'
}

windows = list(set(full_futures['window']))
window = windows[5]
bins = 150   # heatmap resolution per axis

w_df = full_futures[full_futures['window'] == window]
gt = w_df[w_df['model'] == 'ground_truth']
preds = w_df[w_df['model'].isin(models)]

# shared extent so every panel covers the same area
x_edges = np.linspace(preds['x_m'].min(), preds['x_m'].max(), bins + 1)
y_edges = np.linspace(preds['y_m'].min(), preds['y_m'].max(), bins + 1)

# histogram every model first, so all panels share one color scale
hists = {}
for model in models:
    df = preds[preds['model'] == model]
    h, _, _ = np.histogram2d(df['x_m'], df['y_m'], bins=[x_edges, y_edges])
    hists[model] = h / df['future'].nunique()   # per sampled future, so different K compare
norm = LogNorm(vmin=min(h[h > 0].min() for h in hists.values()),
               vmax=max(h.max() for h in hists.values()))

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
for ax, model in zip(axs.flat, ['ground_truth'] + models):
    if model != 'ground_truth':
        # empty bins stay blank instead of the lowest color
        heat = ax.pcolormesh(x_edges, y_edges, np.ma.masked_equal(hists[model], 0).T,
                             cmap='viridis', norm=norm)

    ax.plot(gt['x_m'], gt['y_m'], color='black', linewidth=2)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('X (east)')
    ax.set_ylabel('Y (north)')
    ax.set_title(titles.get(model, 'Ground Truth'), fontsize=14, fontweight='bold')

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.95, wspace=0.05, hspace=0.15)

# shared density colorbar along the bottom
cax = fig.add_axes([0.3, 0.06, 0.4, 0.015])
fig.colorbar(heat, cax=cax, orientation='horizontal',
             label='Predicted points per future in each cell (log scale)')

fig.savefig('heatmap_plot.png')
