import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

full_futures = pd.read_csv('../futures_2_NEW.csv')

models = [
    'ground_truth',
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
    'ground_truth': 'Ground Truth',
    '6_6_128_SWiDiT': 'SWiDiT_6_128',
    '6_4_384_SWiDiT': 'SWiDiT_4_384',
    '6_4_256_SWiDiT': 'SWiDiT_4_256',
    '6_4_128_SWiDiT': 'SWiDiT_4_128',
    '6_2_384_SWiDiT': 'SWiDiT_2_384',
    '6_2_256_SWiDiT': 'SWiDiT_2_256',
    '6_2_128_SWiDiT': 'SWiDiT_2_128'
}

windows = list(set(full_futures['window']))
windows.sort()

window = windows[4] # BAD: 1, GOOD: 4
future = None

def part_futures(model: str, window: int|list[int]|None=None, future: int|list[int]|None=None):
    try:
        retrieve = full_futures[full_futures['model'] == model]

        if type(window) == int:
            retrieve = retrieve[retrieve['window'] == window]
        elif type(window) == list:
            retrieve = retrieve[retrieve['window'] in window]

        if type(future) == int:
            retrieve = retrieve[retrieve['future'] == future]
        elif type(future) == list:
            retrieve = retrieve[retrieve['future'] in future]

        return retrieve

    except Exception as e:
        raise e + '\n One or more arguments are invalid'


futures_list = {
    model: part_futures(model, window, future)
    for model in models
}

print(futures_list)


gt = futures_list['ground_truth']

# shared limits so the ground truth line sits in the same place in every panel
all_df = pd.concat(futures_list.values())
lims = {c: (all_df[c].min(), all_df[c].max()) for c in ('x_m', 'y_m', 'z_m')}
step_norm = plt.Normalize(all_df['step'].min(), all_df['step'].max())

def step_line(ax, df):
    # line whose color changes continuously with the time step
    pts = df[['x_m', 'y_m', 'z_m']].to_numpy()
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap='viridis', norm=step_norm)
    lc.set_array(df['step'].to_numpy()[:-1])
    ax.add_collection(lc)

def avg_ade_fde(df):
    # same as evaluate.py: x/y error in metres, averaged over all sampled futures
    m = df.merge(gt[['step', 'x_m', 'y_m']], on='step', suffixes=('', '_gt'))
    m['err'] = np.hypot(m['x_m'] - m['x_m_gt'], m['y_m'] - m['y_m_gt'])
    ade = m.groupby('future')['err'].mean().mean()
    fde = m[m['step'] == m['step'].max()]['err'].mean()
    return ade, fde

fig, axs = plt.subplots(nrows=2, ncols=4, subplot_kw={'projection': '3d', 'computed_zorder': False}, figsize=(20, 10))
for i, (model, df) in enumerate(futures_list.items()):
    ax = axs[i // 4][i % 4]

    if model != 'ground_truth':
        sc_plot = ax.scatter(
            df['x_m'], df['y_m'], df['z_m'],
            c=df['step'], cmap='viridis', norm=step_norm,
            s=5, label=model
        )

        # one line per sampled future, so separate futures aren't joined together
        for _, f in df.groupby('future'):
            step_line(ax, f)

    ax.plot(
        gt['x_m'], gt['y_m'], gt['z_m'],
        color='black', linewidth=2.5, zorder=4
    )

    ax.set_xlim(*lims['x_m'])
    ax.set_ylim(*lims['y_m'])
    ax.set_zlim(*lims['z_m'])
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
    ax.annotate(titles.get(model, model), xy=(0.5, label_y), xycoords='axes fraction',
                xytext=(0, 14), textcoords='offset points',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    if model != 'ground_truth':
        ade, fde = avg_ade_fde(df)
        ax.text2D(0.5, label_y, f'avgADE {ade:.1f} m   avgFDE {fde:.1f} m',
                  transform=ax.transAxes, ha='center', va='bottom', fontsize=12)

fig.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=0.95,
                    wspace=-0.1, hspace=0.05)

# step colorbar along the bottom: [left, bottom, width, height] in figure fractions
cax = fig.add_axes([0.3, 0.05, 0.4, 0.015])
fig.colorbar(sc_plot, cax=cax, orientation='horizontal', label='Time step')
fig.savefig('single_plot.png')
