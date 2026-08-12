import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D

def projections(repository, arq_name, dt, perp, o3a_intervals):
    final_vecs = np.load(f'{repository}vecs_{arq_name}_dt{str(dt)}.npy')
    final_labs = np.load(f'{repository}labs_{arq_name}_dt{str(dt)}.npy')
    final_GPSs = np.load(f'{repository}GPSs_{arq_name}_dt{str(dt)}.npy')

    tsne = TSNE(n_components=2, n_jobs=-1, verbose=0, perplexity=perp, learning_rate=200, 
            max_iter=1000, metric="euclidean", random_state=0)

    proj = tsne.fit_transform(final_vecs)
    proj_df = pd.DataFrame(proj, columns=['x axis', 'y axis'])
    labs = pd.DataFrame(final_labs).rename(columns={0 : 'vecs'})
    GPSs = pd.DataFrame(final_GPSs).rename(columns={0 : 'GPSs'})
    
    labs = np.array(labs).ravel()
    gps_1d = np.array(GPSs).ravel()

    projs_por_mes = {}

    for m, (t_start, t_end) in o3a_intervals.items():
    
        mask = (gps_1d >= t_start) & (gps_1d < t_end)
    
        projs_por_mes[f"proj_{m.lower()}"] = pd.DataFrame({
        'tsne_1': proj_df.iloc[mask, 0],
        'tsne_2': proj_df.iloc[mask, 1],
        'label': labs[mask],
        'GPStime': gps_1d[mask]
        })

    return projs_por_mes

def plot_saz(projs, title, color=None, xlim=None, ylim=None):

    tam = len(projs)
    n_plots = tam + 1 if tam % 2 != 0 else tam
    ncols = n_plots // 2
    
    fig, axs = plt.subplots(2, ncols, figsize=(3*ncols, 4))
    axs = axs.ravel()
    
    labs = pd.concat([df['label'] for df in projs.values()])
    
    le = LabelEncoder()
    le.fit(labs)
    cmap = plt.get_cmap("tab10")
    
    for i, (key, data) in enumerate(projs.items()):
        
        cores = le.transform(data['label'])

        if color is not None:
            axs[i].scatter(data["tsne_1"], data["tsne_2"], c=color, s=3.0, alpha=0.6)
        else:
            axs[i].scatter(data["tsne_1"], data["tsne_2"], c=cores, cmap=cmap, s=3.0, alpha=0.6)
            
        axs[i].set_title(f'{key} / 2019 - ({len(data)} glitches)', fontsize=10, fontweight='bold')
        axs[i].grid()

        if xlim is not None:
            axs[i].set_xlim(xlim)
        if ylim is not None:
            axs[i].set_ylim(ylim)
            
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color 
            if color is not None else cmap(idx), markersize=8, label=label) for idx, label in enumerate(le.classes_)]

    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.0)
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.2, 0.5), title="Glitches")
    fig.tight_layout(rect=[0, 0, 1.1, 0.95])
    plt.show()
    
def plot_single_glitchgram(unclustered, clustered, target_gps, deltat, fmin=3, fmax=8192, tbins=41, fbins=30):

    idx_closest = (clustered["GPStime"] - target_gps).abs().idxmin()
    row = clustered.loc[idx_closest]
    t_center = row["GPStime"]
    label_glitch = row["label"]

    # print(f"Glitch encontrado! Target GPS: {target_gps} | Encontrado: {t_center} | Tipo: {label_glitch}")

    times_all = unclustered["time"].values
    freqs_all = unclustered["frequency"].values
    snrs_all = unclustered["snr"].values

    tmin = t_center - deltat / 2
    tmax = t_center + deltat / 2

    time_edges = np.linspace(-deltat / 2, deltat / 2, tbins + 1)
    freq_edges = np.logspace(np.log10(fmin), np.log10(fmax), fbins + 1)

    i0 = np.searchsorted(times_all, tmin, side="left")
    i1 = np.searchsorted(times_all, tmax, side="right")

    times = times_all[i0:i1] - t_center
    freqs = freqs_all[i0:i1]
    snrs = snrs_all[i0:i1]

    if len(times) == 0:
        print("Nenhum trigger do Omicron encontrado nessa janela de tempo!")
        return

    t_idx = np.digitize(times, time_edges) - 1
    f_idx = np.digitize(freqs, freq_edges) - 1

    A = np.zeros((fbins, tbins), dtype=float)

    for ti, fi, si in zip(t_idx, f_idx, snrs):
        if 0 <= ti < tbins and 0 <= fi < fbins:
            if si > A[fi, ti]:
                A[fi, ti] = si

    plt.figure(figsize=(10, 6))

    X_mesh, Y_mesh = np.meshgrid(time_edges, freq_edges)
    A_plot = np.where(A == 0, np.nan, A)

    mesh = plt.pcolormesh(X_mesh, Y_mesh, A_plot, cmap="inferno", shading="flat")

    plt.yscale("log")
    plt.ylim(fmin, fmax)
    plt.xlim(-deltat / 2, deltat / 2)

    cbar = plt.colorbar(mesh)
    cbar.set_label("Signal-to-Noise Ratio (SNR)", fontsize=11)

    plt.title(
        f"GSPY: {label_glitch} - GPS: {t_center}",
        fontsize=12,
    )
    plt.xlabel(f"Tempo relativo ao centro [s] (dt = {deltat}s)", fontsize=11)
    plt.ylabel("Frequência [Hz]", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.show()
    

