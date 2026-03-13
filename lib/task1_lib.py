import numpy as np
import matplotlib.pyplot as plt
from lib.kernels import gaussian_kernel, smooth_1d, smooth_3d
from lib.detect_spikes import *

DT = 0.01

def _window_slice(center_idx, rel_window):
    """
    Convert a relative window [start, end] into absolute slice indices.
    Inclusive on both ends conceptually, returned as Python slice bounds [start, end+1).
    """
    rel_start, rel_end = rel_window
    abs_start = center_idx + rel_start
    abs_end = center_idx + rel_end
    return abs_start, abs_end + 1

def _first_settle_index_after_peak(signal, peak_idx, threshold):
    """
    Return first index >= peak_idx where signal <= threshold.
    Return None if not found.
    """
    for k in range(peak_idx, len(signal)):
        if signal[k] <= threshold:
            return k
    return None

def _safe_stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

def print_summary(label, summary):
    print(f"\n--- {label} ---")

    for metric, stats in summary.items():
        print(f"\n{metric}:")
        print(f"  count : {stats['count']}")
        print(f"  mean  : {stats['mean']:.6f}")
        print(f"  std   : {stats['std']:.6f}")
        print(f"  min   : {stats['min']:.6f}")
        print(f"  max   : {stats['max']:.6f}")

def get_impulses_and_spikes(t, com, num_spikes, dt = 0.01, imp_offset=25, plot = False):

    com_pos = np.asarray(com)
    gk_pos = gaussian_kernel(sigma=1, radius=3)
    com_pos_smoothed = smooth_3d(com, gk_pos)
    com_pos_norm = np.linalg.norm(com_pos_smoothed, axis=1)  

    com_vel = np.gradient(com_pos, dt, axis=0)   # shape (N,3)
    gk_vel = gaussian_kernel(sigma=1, radius = 10)
    com_vel_smoothed = smooth_3d(com_vel, gk_vel)
    com_vel_norm = np.linalg.norm(com_vel_smoothed, axis=1)     

    vel_spike_idx, vel_spike_times, vel_spike_vals, props = detect_known_number_of_spikes(
        signal=com_vel_norm,
        timestamps=t,
        num_spikes=num_spikes,
        min_prominence=auto_prominence(com_vel_norm, frac=0.25),
        min_width=4,
        min_distance=20,
    )
    imp_idx = vel_spike_idx - imp_offset

    com_spike_idx, com_spike_times, com_spike_vals, props = detect_known_number_of_spikes(
        signal=com_pos_norm,
        timestamps=t,
        num_spikes=num_spikes,
        min_prominence=auto_prominence(com_pos_norm, frac=0.25),
        min_width=20,
        min_distance=20,
    )

    if plot:
        plt.figure(figsize=(11, 5))
        plt.plot(t, com_vel_norm, label="Smoothed Vel magnitude")
        plt.plot(t, com_pos_norm, label="Smoothed CoM magnitude")
        for idx in imp_idx:
            plt.axvline(t[idx], linestyle="--", alpha=0.4, color="red")
        for idx in com_spike_idx:
            plt.axvline(t[idx], linestyle="--", alpha=0.4, color="blue")
        plt.xlabel("Time [s]")
        plt.ylabel("||CoM - goal|| [m]")
        plt.title("Detected disturbance times")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return imp_idx, com_spike_idx, vel_spike_idx

def analyze_position_data(
    samples,
    impulses,
    spikes,
    baseline_reference_window,
    integral_window,
    settle_time_threshold=1 / 2.0,
    plot = False
):
    """
    Analyze stable-standing position response around each spike.

    Parameters
    ----------
    samples : ndarray, shape (N, 3)
        Full CoM position trajectory over time.
    impulses : ndarray, shape (K,)
        Impulse indices, i.e. time of impulse.
    spikes : ndarray, shape (K,)
        Spike indices, i.e. time of maximal COM extrusion.
    baseline_reference_window : [start, end]
        Relative sample window used to compute baseline mean, e.g. [-200, -100].
    integral_window : [start, end]
        Relative sample window used to compute integral error, e.g. [-100, 200].
    settle_time_threshold : float
        Settling threshold as a fraction of peak. Default 1/e.

    Returns
    -------
    results : list[dict]
        Per-spike metrics.
    summary : dict
        Summary statistics across spikes.
    """
    samples = np.asarray(samples, dtype=float)
    impulses = np.asarray(impulses, dtype=int)
    spikes = np.asarray(spikes, dtype=int)

    results = []

    for i in range(len(impulses)):
        imp_idx = impulses[i]
        spike_idx = spikes[i]
        # Absolute window bounds
        b_start, b_stop = _window_slice(imp_idx, baseline_reference_window)
        i_start, i_stop = _window_slice(imp_idx, integral_window)

        # Skip spikes too close to boundaries
        if b_start < 0 or b_stop > len(samples):
            raise ValueError(f"Baseline window index overflow for impulse at t_index = {imp_idx}")
            continue
        if i_start < 0 or i_stop > len(samples):
            raise ValueError(f"Integral window index overflow for impulse at t_index = {imp_idx}")
            continue

        # Extract windows
        baseline_samples = samples[b_start:b_stop]          # shape (B,3)
        integral_samples = samples[i_start:i_stop]          # shape (I,3)

        # Baseline mean vector from baseline window
        baseline = np.mean(baseline_samples, axis=0)        # shape (3,)
        integral_centered = integral_samples - baseline
        gk = gaussian_kernel(sigma=1, radius=3)
        integral_centered_smoothed = smooth_3d(integral_centered, gk)
        integral_norm = np.linalg.norm(integral_centered_smoothed, axis=1)

        # Local index of spike within analysis window
        peak_local_idx = spike_idx - i_start
        peak_error = float(integral_norm[peak_local_idx])

        # Settling time relative to spike, threshold = threshold * peak
        settle_threshold_value = settle_time_threshold * peak_error
        settle_local_idx = _first_settle_index_after_peak(
            integral_norm, peak_local_idx, settle_threshold_value
        )

        if settle_local_idx is None:
            raise ValueError(f"Signal never settles below threhold for impulse at t_index = {imp_idx}")
        else:
            settling_time = (settle_local_idx - peak_local_idx) * DT

        # Integral over requested window
        integrated_error = float(np.sum(integral_norm) * DT)

        results.append({
            "impulse_idx": int(imp_idx),
            "peak_error": peak_error, #meters
            "settling_time": settling_time, #seconds
            "settling_threshold_value": settle_threshold_value,
            "integrated_error": integrated_error,
            "peak_local_idx": int(peak_local_idx),
            "settle_local_idx": int(settle_local_idx) if settle_local_idx is not None else None,
            "integral_norm": integral_norm,
        })

    summary = {
        "peak_error": _safe_stats([r["peak_error"] for r in results]),
        "settling_time_samples": _safe_stats([r["settling_time"] for r in results]),
        "integrated_error": _safe_stats([r["integrated_error"] for r in results]),
    }

    if plot:
        plt.figure(figsize=(11, 5))
        window_t = np.arange(integral_window[0], integral_window[1] + 1) * DT

        for k, r in enumerate(results):
            y = r["integral_norm"]
            plt.plot(window_t, y, alpha=0.7, label=f"Spike {k+1}")

            plt.scatter(
                window_t[r["peak_local_idx"]],
                y[r["peak_local_idx"]],
                marker="o"
            )

            if r["settle_local_idx"] is not None:
                plt.scatter(
                    window_t[r["settle_local_idx"]],
                    y[r["settle_local_idx"]],
                    marker="x"
                )

            plt.axhline(r["settling_threshold_value"], linestyle="--", alpha=0.3)

        plt.axvline(0.0, linestyle="--", color="k", alpha=0.6)
        plt.xlabel("Time relative to impulse [s]")
        plt.ylabel("||CoM - baseline|| [m]")
        plt.title("Event-aligned CoM deviation norm")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results, summary

def analyze_velocity_data(
    samples,
    impulses,
    spikes,
    integral_window,
    plot=False,
):
    """
    Analyze stable-standing velocity response around each spike.

    Parameters
    ----------
    samples : ndarray, shape (N, 3)
        Full CoM position trajectory over time.
    impulses : ndarray, shape (K,)
        Impulse indices, i.e. time of impulse.
    spikes : ndarray, shape (K,)
        Spike indices, i.e. time of maximal COM extrusion.
    baseline_reference_window : [start, end]
        Relative sample window used to compute baseline mean, e.g. [-200, -100].
    integral_window : [start, end]
        Relative sample window used to compute integral error, e.g. [-100, 200].

    Returns
    -------
    results : list[dict]
        Per-spike metrics.
    summary : dict
        Summary statistics across spikes.
    """
    vels = np.gradient(samples, DT, axis=0)   # shape (N,3)
    impulses = np.asarray(impulses, dtype=int)
    spikes = np.asarray(spikes, dtype=int)

    results = []

    for i in range(len(impulses)):
        imp_idx = impulses[i]
        spike_idx = spikes[i]
        # Absolute window bounds
        i_start, i_stop = _window_slice(imp_idx, integral_window)

        # Skip spikes too close to boundaries
        if i_start < 0 or i_stop > len(samples):
            raise ValueError(f"Integral window index overflow for impulse at t_index = {imp_idx}")
            continue

        # Extract windows
        integral_vels = vels[i_start:i_stop]          # shape (I,3)

        integral_kinetic_energy = np.sum(integral_vels**2, axis=1)
        gk = gaussian_kernel(sigma=3, radius=10)
        integral_kinetic_energy_smoothed = smooth_1d(integral_kinetic_energy, gk)

        # Local index of spike within analysis window
        peak_local_idx = spike_idx - i_start
        peak_speed = float(np.sqrt(integral_kinetic_energy_smoothed[peak_local_idx]))

        # Integral over requested window
        integrated_kinetic_energy = float(np.sum(integral_kinetic_energy_smoothed) * DT)

        results.append({
            "impulse_idx": int(imp_idx),
            "peak_speed": peak_speed, #meters
            "integrated_kinetic_energy": integrated_kinetic_energy,
            "integral_kinetic_energy_smoothed": integral_kinetic_energy_smoothed,
            "peak_local_idx": int(peak_local_idx),
        })

    summary = {
        "peak_speed": _safe_stats([r["peak_speed"] for r in results]),
        "integrated_kinetic_energy": _safe_stats([r["integrated_kinetic_energy"] for r in results]),
    }

    if plot:
        # 3) Overlay event-aligned ||v||^2
        plt.figure(figsize=(11, 5))
        for k, r in enumerate(results):
            window_t = np.arange(integral_window[0], integral_window[1] + 1) * DT
            y = r["integral_kinetic_energy_smoothed"]
            plt.plot(window_t, y, alpha=0.7, label=f"Spike {k+1}")

            plt.scatter(
                window_t[r["peak_local_idx"]],
                y[r["peak_local_idx"]],
                marker="o"
            )
        plt.axvline(0.0, linestyle="--", color="k", alpha=0.6)
        plt.xlabel("Time relative to spike [s]")
        plt.ylabel("||v||^2")
        plt.title("Event-aligned velocity norm squared")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results, summary

def plot_metric_summary(metric_name, summaries, trial_labels, ylabel=None):
    """
    summaries: list of summary dicts, one per trial
    metric_name: key inside each summary dict
    trial_labels: e.g. ["tran", "imp", "mpc"]
    """
    means = [s[metric_name]["mean"] for s in summaries]
    stds  = [s[metric_name]["std"]  for s in summaries]

    x = np.arange(len(trial_labels))

    plt.figure(figsize=(7, 5))
    plt.bar(x, means, yerr=stds, capsize=6, alpha=0.8)
    plt.xticks(x, trial_labels)
    plt.ylabel(ylabel if ylabel is not None else metric_name)
    plt.title(metric_name.replace("_", " ").title())
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()