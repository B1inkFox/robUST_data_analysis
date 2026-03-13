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
    settle_time_threshold=1 / np.e,
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
    impulses = np.asarray(impulses, dtype=float)
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
            raise ValueError("Baseline window index overflow for impulse at t_index = ", imp_idx)
            continue
        if i_start < 0 or i_stop > len(samples):
            raise ValueError("Integral window index overflow for impulse at t_index = ", imp_idx)
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
        peak_local_idx = spike_idx - imp_idx
        peak_error = float(integral_norm[peak_local_idx])

        # Settling time relative to spike, threshold = threshold * peak
        settle_threshold_value = settle_time_threshold * peak_error
        settle_local_idx = _first_settle_index_after_peak(
            integral_norm, peak_local_idx, settle_threshold_value
        )

        if settle_local_idx is None:
            raise ValueError("Signal never settles below threhold for impulse at t_index = ", imp_idx)
        else:
            settling_time = (settle_local_idx - peak_local_idx) * DT

        # Integral over requested window
        integrated_error = float(np.sum(integral_norm))

        results.append({
            "impulse_idx": int(imp_idx),
            "peak_error": peak_error, #meters
            "settling_time": settling_time, #seconds
            "settling_threshold_value": settle_threshold_value,
            "integrated_error": integrated_error,
        })

    summary = {
        "peak_error": _safe_stats([r["peak_error"] for r in results]),
        "settling_time_samples": _safe_stats([r["settling_time"] for r in results]),
        "integrated_error": _safe_stats([r["integrated_error"] for r in results]),
    }

    return results, summary

def analyze_velocity_data(
    samples,
    impulses,
    spikes,
    baseline_reference_window,
    integral_window,
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
    impulses = np.asarray(impulses, dtype=float)
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
            raise ValueError("Baseline window index overflow for impulse at t_index = ", imp_idx)
            continue
        if i_start < 0 or i_stop > len(samples):
            raise ValueError("Integral window index overflow for impulse at t_index = ", imp_idx)
            continue

        # Extract windows
        baseline_vels = vels[b_start:b_stop]          # shape (B,3)
        integral_vels = vels[i_start:i_stop]          # shape (I,3)

        baseline_kinetic_energy = np.sum(baseline_vels**2, axis=1)
        baseline_kinetic_energy_average = np.mean(baseline_kinetic_energy, axis=0)        # shape (3,)
        integral_kinetic_energy = np.sum(integral_vels**2, axis=1)
        integral_kinetic_energy_centered = integral_kinetic_energy - baseline_kinetic_energy_average
        gk = gaussian_kernel(sigma=1, radius=10)
        integral_kinetic_energy_smoothed = smooth_3d(integral_kinetic_energy_centered, gk)

        # Local index of spike within analysis window
        peak_local_idx = spike_idx - imp_idx
        peak_speed = float(np.sqrt(integral_kinetic_energy_smoothed[peak_local_idx]))

        # Integral over requested window
        integrated_kinetic_energy = float(np.sum(integral_kinetic_energy_smoothed))

        results.append({
            "impulse_idx": int(imp_idx),
            "peak_speed": peak_speed, #meters
            "integrated_kinetic_energy": integrated_kinetic_energy,
        })

    summary = {
        "peak_speed": _safe_stats([r["peak_speed"] for r in results]),
        "integrated_kinetic_energy": _safe_stats([r["integrated_kinetic_energy"] for r in results]),
    }

    return results, summary