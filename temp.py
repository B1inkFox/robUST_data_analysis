import numpy as np

DT = 0.01

def _smooth_signal_1d(signal, smoothing_kernel=None):
    """
    Smooth a 1D signal with a convolution kernel.
    If smoothing_kernel is None, return a copy of the input.
    """
    signal = np.asarray(signal, dtype=float)

    if smoothing_kernel is None:
        return signal.copy()

    kernel = np.asarray(smoothing_kernel, dtype=float)
    return np.convolve(signal, kernel, mode="same")


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


def analyze_spike_window_stable_standing_position(
    samples,
    spikes,
    baseline_reference_window,
    integral_window,
    smoothing_kernel=None,
    settle_time_threshold=1 / np.e,
    analysis_window=None,
):
    """
    Analyze stable-standing position response around each spike.

    Parameters
    ----------
    samples : ndarray, shape (N, 3)
        Full CoM position trajectory over time.
    spikes : ndarray, shape (K,)
        Spike indices.
    baseline_reference_window : [start, end]
        Relative sample window used to compute baseline mean, e.g. [-200, -100].
    integral_window : [start, end]
        Relative sample window used to compute integral error, e.g. [-100, 200].
    smoothing_kernel : ndarray or None
        Optional 1D kernel applied to the norm signal after baseline subtraction.
    settle_time_threshold : float
        Settling threshold as a fraction of peak. Default 1/e.
    analysis_window : [start, end] or None
        Relative sample window over which peak / settling are evaluated.
        If None, uses the union of baseline and integral windows plus spike sample 0.

    Returns
    -------
    results : list[dict]
        Per-spike metrics.
    summary : dict
        Summary statistics across spikes.
    """
    samples = np.asarray(samples, dtype=float)
    spikes = np.asarray(spikes, dtype=int)

    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("samples must have shape (N, 3) for 3D CoM position data.")

    if analysis_window is None:
        w_start = min(baseline_reference_window[0], integral_window[0], 0)
        w_end = max(baseline_reference_window[1], integral_window[1], 0)
        analysis_window = [w_start, w_end]

    results = []

    for spike_idx in spikes:
        # Absolute window bounds
        a_start, a_stop = _window_slice(spike_idx, analysis_window)
        b_start, b_stop = _window_slice(spike_idx, baseline_reference_window)
        i_start, i_stop = _window_slice(spike_idx, integral_window)

        # Skip spikes too close to boundaries
        if a_start < 0 or a_stop > len(samples):
            continue
        if b_start < 0 or b_stop > len(samples):
            continue
        if i_start < 0 or i_stop > len(samples):
            continue

        # Extract windows
        analysis_samples = samples[a_start:a_stop]          # shape (W,3)
        baseline_samples = samples[b_start:b_stop]          # shape (B,3)
        integral_samples = samples[i_start:i_stop]          # shape (I,3)

        # Baseline mean vector from baseline window
        baseline = np.mean(baseline_samples, axis=0)        # shape (3,)

        # Subtract baseline from analysis / integral windows
        analysis_centered = analysis_samples - baseline
        integral_centered = integral_samples - baseline

        # Norm of deviation
        analysis_norm = np.linalg.norm(analysis_centered, axis=1)
        integral_norm = np.linalg.norm(integral_centered, axis=1)

        # Optional smoothing on norm
        analysis_norm_smoothed = _smooth_signal_1d(analysis_norm, smoothing_kernel)
        integral_norm_smoothed = _smooth_signal_1d(integral_norm, smoothing_kernel)

        # Local index of spike within analysis window
        spike_local_idx = -analysis_window[0]

        # Peak error after spike
        post_signal = analysis_norm_smoothed[spike_local_idx:]
        peak_local_post = int(np.argmax(post_signal))
        peak_local_idx = spike_local_idx + peak_local_post
        peak_error = float(analysis_norm_smoothed[peak_local_idx])

        # Settling time relative to spike, threshold = threshold * peak
        settle_threshold_value = settle_time_threshold * peak_error
        settle_local_idx = _first_settle_index_after_peak(
            analysis_norm_smoothed, peak_local_idx, settle_threshold_value
        )

        if settle_local_idx is None:
            settling_time_samples = np.nan
        else:
            settling_time_samples = settle_local_idx - spike_local_idx

        # Integral over requested window
        integrated_error = float(np.sum(integral_norm_smoothed))

        results.append({
            "spike_idx": int(spike_idx),
            "baseline": baseline,
            "peak_error": peak_error,
            "settling_time_samples": settling_time_samples,
            "settling_threshold_value": settle_threshold_value,
            "integrated_error": integrated_error,
        })

    summary = {
        "peak_error": _safe_stats([r["peak_error"] for r in results]),
        "settling_time_samples": _safe_stats([r["settling_time_samples"] for r in results]),
        "integrated_error": _safe_stats([r["integrated_error"] for r in results]),
    }

    return results, summary


def analyze_spike_window_stable_standing_velocity(
    samples,
    spikes,
    integral_window,
    smoothing_kernel=None,
    settle_time_threshold=1 / np.e,
    analysis_window=None,
):
    """
    Analyze stable-standing velocity response around each spike.

    Parameters
    ----------
    samples : ndarray, shape (N,)
        Full 1D signal over time, intended to be velocity norm or velocity norm squared.
    spikes : ndarray, shape (K,)
        Spike indices.
    integral_window : [start, end]
        Relative sample window used to compute integral error, e.g. [-100, 200].
    smoothing_kernel : ndarray or None
        Optional 1D kernel applied to the signal.
    settle_time_threshold : float
        Settling threshold as a fraction of peak. Default 1/e.
    analysis_window : [start, end] or None
        Relative sample window over which peak / settling are evaluated.
        If None, uses integral_window expanded to include spike sample 0.

    Returns
    -------
    results : list[dict]
        Per-spike metrics.
    summary : dict
        Summary statistics across spikes.
    """
    samples = np.asarray(samples, dtype=float)
    spikes = np.asarray(spikes, dtype=int)

    if samples.ndim != 1:
        raise ValueError("samples must be 1D for velocity analysis.")

    if analysis_window is None:
        w_start = min(integral_window[0], 0)
        w_end = max(integral_window[1], 0)
        analysis_window = [w_start, w_end]

    results = []

    for spike_idx in spikes:
        a_start, a_stop = _window_slice(spike_idx, analysis_window)
        i_start, i_stop = _window_slice(spike_idx, integral_window)

        if a_start < 0 or a_stop > len(samples):
            continue
        if i_start < 0 or i_stop > len(samples):
            continue

        analysis_signal = samples[a_start:a_stop]
        integral_signal = samples[i_start:i_stop]

        # Baseline is defined as zero
        analysis_processed = _smooth_signal_1d(analysis_signal, smoothing_kernel)
        integral_processed = _smooth_signal_1d(integral_signal, smoothing_kernel)

        spike_local_idx = -analysis_window[0]

        # Peak after spike
        post_signal = analysis_processed[spike_local_idx:]
        peak_local_post = int(np.argmax(post_signal))
        peak_local_idx = spike_local_idx + peak_local_post
        peak_error = float(analysis_processed[peak_local_idx])

        # Settling time to threshold * peak
        settle_threshold_value = settle_time_threshold * peak_error
        settle_local_idx = _first_settle_index_after_peak(
            analysis_processed, peak_local_idx, settle_threshold_value
        )

        if settle_local_idx is None:
            settling_time_samples = np.nan
        else:
            settling_time_samples = settle_local_idx - spike_local_idx

        # Integral error, baseline is zero
        integrated_error = float(np.sum(integral_processed))

        results.append({
            "spike_idx": int(spike_idx),
            "peak_error": peak_error,
            "settling_time_samples": settling_time_samples,
            "settling_threshold_value": settle_threshold_value,
            "integrated_error": integrated_error,
        })

    summary = {
        "peak_error": _safe_stats([r["peak_error"] for r in results]),
        "settling_time_samples": _safe_stats([r["settling_time_samples"] for r in results]),
        "integrated_error": _safe_stats([r["integrated_error"] for r in results]),
    }

    return results, summary