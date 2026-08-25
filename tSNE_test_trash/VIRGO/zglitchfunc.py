import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def glitch_to_glitchgram(unclustered, clustered, savepath, arq_name, deltat, 
                         fmin=3, fmax=8192, tbins=41, fbins=30):
    """
    This function create one glitchgram from unclustered Omicron data, for each glitch.

    Parameters:
        df_triggers (dataFrame): A dataFrame of GPStimes of all triggers inside of some window.
        tmin, tmax (numbers): limits of GPStimes that contain all triggers of some glitch.
        fmin, fmax (numbers): frequency range for the glitchgram.
        n_time_bins (int): number of time bins.
        n_freq_bins (int): number of frequency bins.
        norm (boolean): True if you need normalize SNR values.

    Returns:
        array for some glitch: each entry is a SNR value from flattened (binsf × binst) matrix.
    """

    times_all = unclustered['time'].values
    freqs_all = unclustered['frequency'].values
    snrs_all  = unclustered['snr'].values

    vecs = []
    labs = []
    GPSs = []

    time_edges = np.linspace(-deltat/2, deltat/2, tbins+1)
    freq_edges = np.logspace(np.log10(fmin), np.log10(fmax), fbins+1)
    
    for i, row in enumerate(clustered.itertuples(index=False)):

        #t_center = getattr(row, 'GPStime')
        t_center = row.GPStime
    
        tmin = t_center - deltat/2
        tmax = t_center + deltat/2
    
        i0 = np.searchsorted(times_all, tmin, side="left")
        i1 = np.searchsorted(times_all, tmax, side="right")
    
        times = times_all[i0:i1] - t_center
        freqs = freqs_all[i0:i1]
        snrs  = snrs_all[i0:i1]

        if len(times) < 10: # Equivalente ao tcut
            continue
    
        df_triggers = pd.DataFrame({'time': times,'frequency': freqs,'snr': snrs})
        
        t_idx = np.digitize(df_triggers['time'].values, time_edges) - 1
        f_idx = np.digitize(df_triggers['frequency'].values, freq_edges) - 1
            
        # initial empty matrix
        A = np.zeros((fbins, tbins), dtype=float)
    
        for ti, fi, si in zip(t_idx, f_idx, snrs):
            if 0 <= ti < tbins and 0 <= fi < fbins:
                if si > A[fi, ti]:
                    A[fi, ti] = si
                    
        '''# normalização do SNR 
        if len(snrs) > 0:
            snr_min = snrs.min()
            snr_max = snrs.max()
            if snr_max > snr_min:
                norm_snrs = (snrs - snr_min) / (snr_max - snr_min)
            else:
                norm_snrs = np.zeros_like(snrs)
        else:
            norm_snrs = snrs'''
        
        vecs.append(A.flatten(order="C"))
        labs.append(clustered['label'][i])
        GPSs.append(clustered['GPStime'][i])

    vecs_final = np.array(vecs).astype('float32')
    labs_final = np.array(labs)
    GPSs_final = np.array(GPSs).astype('float32')
    
    np.save(savepath + f'vecs_{arq_name}_dt{deltat}.npy', vecs_final)
    np.save(savepath + f'labs_{arq_name}_dt{deltat}.npy', labs_final)
    np.save(savepath + f'GPSs_{arq_name}_dt{deltat}.npy', GPSs_final)
    
    return vecs_final, labs_final, GPSs_final

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
    

