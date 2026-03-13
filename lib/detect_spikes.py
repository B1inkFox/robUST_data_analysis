import numpy as np
from scipy.signal import find_peaks, peak_widths


def detect_known_number_of_spikes(
    signal,
    timestamps,
    num_spikes,
    min_prominence=None,
    min_width=None,
    min_distance=None,
):
    """
    Detect exactly `num_spikes` upward spikes in a drifting baseline signal.

    Parameters
    ----------
    signal : array-like, shape (N,)
        Smoothed 1D signal.
    timestamps : array-like, shape (N,)
        Corresponding timestamps.
    num_spikes : int
        Exact number of spikes expected.
    min_prominence : float or None
        Minimum prominence threshold.
    min_width : float or None
        Minimum width in samples.
    min_distance : int or None
        Minimum spacing between neighboring peaks in samples.

    Returns
    -------
    peak_idx : ndarray
        Indices of detected spikes, sorted in time.
    peak_times : ndarray
        Exact timestamps of detected spikes.
    peak_values : ndarray
        Signal values at the detected spikes.
    props : dict
        Peak properties returned by scipy.
    """
    signal = np.asarray(signal)
    timestamps = np.asarray(timestamps)

    peaks, props = find_peaks(
        signal,
        prominence=min_prominence,
        width=min_width,
        distance=min_distance,
    )

    if len(peaks) < num_spikes:
        raise ValueError(
            f"Only found {len(peaks)} candidate peaks, but num_spikes={num_spikes}. "
            f"Try lowering min_prominence/min_width/min_distance."
        )

    # Rank by prominence, not absolute height
    prominences = props["prominences"]
    order = np.argsort(prominences)[::-1]
    selected = peaks[order[:num_spikes]]

    # Sort chronologically
    selected = np.sort(selected)

    peak_times = timestamps[selected]
    peak_values = signal[selected]

    return selected, peak_times, peak_values, props


def auto_prominence(signal, frac=0.2):
    """
    A rough automatic prominence scale.
    Uses a fraction of the signal's robust range.
    """
    signal = np.asarray(signal)
    q10, q90 = np.percentile(signal, [10, 90])
    return frac * (q90 - q10)