import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

full_futures = pd.read_csv('../futures_2_NEW.csv')

models = [
    '6_6_128_SWiDiT',
    '6_4_384_SWiDiT',
    '6_4_256_SWiDiT',
    '6_4_128_SWiDiT',
    '6_2_384_SWiDiT',
    '6_2_256_SWiDiT',
    '6_2_128_SWiDiT'
]

# custom legend names; any model left out falls back to its name
titles = {
    '6_6_128_SWiDiT': 'SWiDiT_6_128',
    '6_4_384_SWiDiT': 'SWiDiT_4_384',
    '6_4_256_SWiDiT': 'SWiDiT_4_256',
    '6_4_128_SWiDiT': 'SWiDiT_4_128',
    '6_2_384_SWiDiT': 'SWiDiT_2_384',
    '6_2_256_SWiDiT': 'SWiDiT_2_256',
    '6_2_128_SWiDiT': 'SWiDiT_2_128'
}

# one unique color per model
colors = {model: plt.cm.tab10(i) for i, model in enumerate(models)}

windows = list(set(full_futures['window']))
windows.sort()
samples = {'Poor Forecast': windows[1], 'Strong Forecast': windows[4]}

def min_fde_future(df, gt):
    # the sampled future whose last step is closest (x/y, as in evaluate.py) to the ground truth's
    last = df[df['step'] == df['step'].max()]
    gt_last = gt[gt['step'] == gt['step'].max()].iloc[0]
    fde = np.hypot(last['x_m'] - gt_last['x_m'], last['y_m'] - gt_last['y_m'])
    best = last['future'].iloc[fde.argmin()]
    return df[df['future'] == best], fde.min()

fig, axs = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d', 'computed_zorder': False}, figsize=(12, 6))
for ax, (name, window) in zip(axs, samples.items()):
    w_df = full_futures[full_futures['window'] == window]
    gt = w_df[w_df['model'] == 'ground_truth']

    best = {}
    for model in models:
        best[model], _ = min_fde_future(w_df[w_df['model'] == model], gt)
        ax.plot(
            best[model]['x_m'], best[model]['y_m'], best[model]['z_m'],
            color=colors[model], linewidth=1.5, zorder=1
        )

    ax.plot(
        gt['x_m'], gt['y_m'], gt['z_m'],
        color='black', linewidth=2.5, zorder=4
    )

    # limits from the ground truth and the plotted futures only
    shown = pd.concat([gt, *best.values()])
    ax.set_xlim(shown['x_m'].min(), shown['x_m'].max())
    ax.set_ylim(shown['y_m'].min(), shown['y_m'].max())
    ax.set_zlim(shown['z_m'].min(), shown['z_m'].max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False

    ax.set_box_aspect(None, zoom=1.0)

    ax.view_init(elev=25, azim=45)

    ax.set_xlabel('X', labelpad=-12)
    ax.set_ylabel('Y', labelpad=-12)
    ax.set_zlabel('Z', labelpad=-12)
    # 3D axes leave empty space above the box; label_y sits the text right on top of it
    label_y = 0.91
    ax.annotate(name, xy=(0.5, label_y), xycoords='axes fraction',
                xytext=(0, 14), textcoords='offset points',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text2D(0.5, label_y, f'Window {window}',
              transform=ax.transAxes, ha='center', va='bottom', fontsize=12)

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.12, top=0.95, wspace=-0.1)

# model color legend along the bottom
handles = [Line2D([], [], color='black', linewidth=2.5)] + \
          [Line2D([], [], color=colors[model], linewidth=2) for model in models]
labels = ['Ground Truth'] + [titles.get(model, model) for model in models]
# top of the legend sits just under the bottom corner of the 3D boxes
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.17),
           ncol=4, frameon=False, fontsize=11)

# crop the empty margin left below the legend
fig.savefig('forecast_types_plot.png', bbox_inches='tight', pad_inches=0.1)
