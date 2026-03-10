import numpy as np
import matplotlib.pyplot as plt

from lib.extract_trial_data import extract_kinematics_and_forceplates, extract_goals
from lib.kernels import gaussian_kernel
from lib.detect_spikes import detect_known_number_of_spikes, auto_prominence


# ============================================================
# CONFIG
# ============================================================
csv_path_tran = "Darren_grfmpc_experiment/Darren_Standing_Tran_1_2026-02-24_11-27-20.csv"
csv_path_imp  = "Darren_grfmpc_experiment/Darren_Standing_Imp_1_2026-02-24_12-12-11.csv"
csv_path_mpc  = "Darren_grfmpc_experiment/Darren_Standing_MPC_1_2026-02-24_11-29-49.csv"

csv_path = csv_path_tran   # choose which file to analyze

dt = 0.01                  # fixed sampling period [s]

PRE_SAMPLES  = 200         # window start relative to spike
POST_SAMPLES = 500         # window end relative to spike

BASELINE_START_REL = -200
BASELINE_END_REL   = -100

INTEGRAL_START_REL = -100
INTEGRAL_END_REL   =  200

NUM_SPIKES = 10


# ============================================================
# HELPERS
# ============================================================
def smooth_1d(x, kernel):
    return np.convolve(x, kernel, mode='same')


def smooth_3d(x, kernel):
    """
    x: (N,3)
    smooth each coordinate separately
    """
    out = np.zeros_like(x)
    for j in range(3):
        out[:, j] = np.convolve(x[:, j], kernel, mode='same')
    return out


def rel_to_local_index(rel_sample, pre_samples):
    """
    In a window [spike-pre_samples, spike+post_samples],
    convert relative sample index to local array index.
    Example: rel=-200 -> 0, rel=0 -> pre_samples
    """
    return rel_sample + pre_samples


def first_below_threshold_after_peak(signal, peak_local_idx, threshold):
    """
    Returns the first local index >= peak_local_idx for which signal <= threshold.
    Returns None if never found.
    """
    for k in range(peak_local_idx, len(signal)):
        if signal[k] <= threshold:
            return k
    return None


def summarize_metric(name, values):
    arr = np.asarray(values, dtype=float)
    print(f"{name}:")
    print(f"  mean = {np.mean(arr):.6f}")
    print(f"  std  = {np.std(arr, ddof=1):.6f}" if len(arr) > 1 else f"  std  = 0.000000")
    print(f"  min  = {np.min(arr):.6f}")
    print(f"  max  = {np.max(arr):.6f}")
    print("")


def analyze_spike_window(window_com, window_vel2, dt, pre_samples):
    """
    window_com  : (W,3), smoothed CoM error trajectory in spike-centered window
    window_vel2 : (W,), ||v||^2 in same window
    pre_samples : number of samples before spike in the window

    Returns dict of metrics.
    """

    # ------------------------------------------------------------
    # Build local index ranges
    # ------------------------------------------------------------
    b0 = rel_to_local_index(BASELINE_START_REL, pre_samples)   # -200
    b1 = rel_to_local_index(BASELINE_END_REL,   pre_samples)   # -100
    # Use inclusive interval [-200, -100], so Python slice is [b0 : b1+1]
    baseline_slice = slice(b0, b1 + 1)

    i0 = rel_to_local_index(INTEGRAL_START_REL, pre_samples)   # -100
    i1 = rel_to_local_index(INTEGRAL_END_REL,   pre_samples)   # +200
    integral_slice = slice(i0, i1 + 1)

    spike_local_idx = pre_samples
    post_slice = slice(spike_local_idx, len(window_com))

    # ------------------------------------------------------------
    # Baseline CoM position
    # ------------------------------------------------------------
    baseline_pos = np.mean(window_com[baseline_slice, :], axis=0)

    # Norm of raw CoM error
    com_norm = np.linalg.norm(window_com, axis=1)

    # Deviation from baseline position
    com_dev_vec = window_com - baseline_pos
    com_dev_norm = np.linalg.norm(com_dev_vec, axis=1)

    # Baseline norm for thresholding on absolute norm
    baseline_norm = np.mean(com_norm[baseline_slice])

    # ------------------------------------------------------------
    # Peak CoM excursion in L2 norm
    # ------------------------------------------------------------
    post_dev_norm = com_dev_norm[post_slice]
    peak_excursion = np.max(post_dev_norm)
    peak_excursion_local_idx = spike_local_idx + np.argmax(post_dev_norm)

    # Settling threshold on absolute CoM norm:
    # baseline_norm + (peak_norm - baseline_norm)/e
    peak_norm_post = np.max(com_norm[post_slice])
    com_threshold = baseline_norm + (peak_norm_post - baseline_norm) / np.e

    com_settle_local_idx = first_below_threshold_after_peak(
        com_norm, peak_excursion_local_idx, com_threshold
    )
    if com_settle_local_idx is None:
        com_settle_time = np.nan
    else:
        com_settle_time = (com_settle_local_idx - spike_local_idx) * dt

    # ------------------------------------------------------------
    # Integrated CoM deviation over [-100, +200]
    # ------------------------------------------------------------
    # Integrate deviation-from-baseline norm
    integrated_com_deviation = np.sum(com_dev_norm[integral_slice]) * dt

    # ------------------------------------------------------------
    # Velocity norm squared
    # ------------------------------------------------------------
    vel2_baseline = np.mean(window_vel2[baseline_slice])
    vel2_post = window_vel2[post_slice]

    peak_vel2 = np.max(vel2_post)
    peak_vel2_local_idx = spike_local_idx + np.argmax(vel2_post)

    vel2_threshold = vel2_baseline + (peak_vel2 - vel2_baseline) / np.e

    vel2_settle_local_idx = first_below_threshold_after_peak(
        window_vel2, peak_vel2_local_idx, vel2_threshold
    )
    if vel2_settle_local_idx is None:
        vel2_settle_time = np.nan
    else:
        vel2_settle_time = (vel2_settle_local_idx - spike_local_idx) * dt

    return {
        "baseline_pos": baseline_pos,
        "baseline_norm": baseline_norm,
        "peak_excursion": peak_excursion,
        "com_settle_time": com_settle_time,
        "integrated_com_deviation": integrated_com_deviation,
        "vel2_baseline": vel2_baseline,
        "peak_vel2": peak_vel2,
        "vel2_settle_time": vel2_settle_time,
        "com_threshold": com_threshold,
        "vel2_threshold": vel2_threshold,
        "com_norm": com_norm,
        "com_dev_norm": com_dev_norm,
    }


# ============================================================
# LOAD DATA
# ============================================================
timestamps, CoM, EE, FP1, FP2 = extract_kinematics_and_forceplates(csv_path)
timestamps, goal_force, goal_torque, goal_position, goal_euler, goal_velocity, goal_angular_velocity = extract_goals(csv_path)

t = np.array(timestamps)

# CoM error relative to goal position
com = np.array([T[0:3, 3] for T in CoM]) - np.array(goal_position)

# Smooth 3D CoM coordinates
gk = gaussian_kernel(sigma=1, radius=3)
com_smoothed = smooth_3d(com, gk)

# Smoothed CoM magnitude for spike detection
com_magnitude_smoothed = np.linalg.norm(com_smoothed, axis=1)

# Detect spikes
peak_idx, peak_times, peak_vals, props = detect_known_number_of_spikes(
    signal=com_magnitude_smoothed,
    timestamps=t,
    num_spikes=NUM_SPIKES,
    min_prominence=auto_prominence(com_magnitude_smoothed, frac=0.25),
    min_width=20,
    min_distance=20,
)

print("Detected spike times:")
print(peak_times)
print("")


# ============================================================
# VELOCITY FROM SMOOTHED CoM
# ============================================================
# np.gradient uses central difference in the interior
com_velocity = np.gradient(com_smoothed, dt, axis=0)   # shape (N,3)
vel2 = np.sum(com_velocity**2, axis=1)                 # ||v||^2


# ============================================================
# ANALYZE EACH SPIKE
# ============================================================
results = []

valid_peak_idx = []
valid_peak_times = []

window_len = PRE_SAMPLES + POST_SAMPLES + 1

for spike_i, idx in enumerate(peak_idx):
    start = idx - PRE_SAMPLES
    stop = idx + POST_SAMPLES + 1   # end-exclusive

    # skip spikes too close to boundaries
    if start < 0 or stop > len(t):
        print(f"Skipping spike at index {idx} (time {t[idx]:.3f}s): window exceeds data bounds.")
        continue

    window_t = t[start:stop]
    window_com = com_smoothed[start:stop, :]
    window_vel2 = vel2[start:stop]

    metrics = analyze_spike_window(
        window_com=window_com,
        window_vel2=window_vel2,
        dt=dt,
        pre_samples=PRE_SAMPLES,
    )

    metrics["global_peak_idx"] = idx
    metrics["global_peak_time"] = t[idx]
    metrics["window_t"] = window_t - t[idx]   # center time axis at spike
    metrics["window_com"] = window_com
    metrics["window_vel2"] = window_vel2

    results.append(metrics)
    valid_peak_idx.append(idx)
    valid_peak_times.append(t[idx])


# ============================================================
# REPORT PER-SPIKE
# ============================================================
print("Per-spike metrics")
print("=" * 60)
for k, r in enumerate(results):
    bx, by, bz = r["baseline_pos"]
    print(f"Spike {k+1}: t = {r['global_peak_time']:.3f} s")
    print(f"  baseline_pos              = [{bx:.6f}, {by:.6f}, {bz:.6f}]")
    print(f"  baseline_norm             = {r['baseline_norm']:.6f}")
    print(f"  peak_excursion            = {r['peak_excursion']:.6f}")
    print(f"  CoM settling time         = {r['com_settle_time']:.6f} s")
    print(f"  integrated CoM deviation  = {r['integrated_com_deviation']:.6f}")
    print(f"  vel2 baseline             = {r['vel2_baseline']:.6f}")
    print(f"  peak vel2                 = {r['peak_vel2']:.6f}")
    print(f"  vel2 settling time        = {r['vel2_settle_time']:.6f} s")
    print("")


# ============================================================
# SUMMARY STATISTICS
# ============================================================
peak_excursion_all = [r["peak_excursion"] for r in results]
com_settle_all = [r["com_settle_time"] for r in results if not np.isnan(r["com_settle_time"])]
integrated_dev_all = [r["integrated_com_deviation"] for r in results]
peak_vel2_all = [r["peak_vel2"] for r in results]
vel2_settle_all = [r["vel2_settle_time"] for r in results if not np.isnan(r["vel2_settle_time"])]

print("")
print("Summary statistics")
print("=" * 60)
summarize_metric("Peak CoM excursion", peak_excursion_all)
summarize_metric("CoM settling time [s]", com_settle_all)
summarize_metric("Integrated CoM deviation", integrated_dev_all)
summarize_metric("Peak ||v||^2", peak_vel2_all)
summarize_metric("Velocity settling time [s]", vel2_settle_all)


# ============================================================
# PLOTS
# ============================================================
# 1) Full trial with detected spikes
plt.figure(figsize=(11, 5))
plt.plot(t, com_magnitude_smoothed, label="Smoothed CoM magnitude")
for idx in valid_peak_idx:
    plt.axvline(t[idx], linestyle="--", alpha=0.4, color="red")
plt.xlabel("Time [s]")
plt.ylabel("||CoM - goal|| [m]")
plt.title("Detected disturbance times")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 2) Overlay event-aligned CoM deviation norm
plt.figure(figsize=(11, 5))
for k, r in enumerate(results):
    plt.plot(r["window_t"], r["com_dev_norm"], alpha=0.7, label=f"Spike {k+1}")
plt.axvline(0.0, linestyle="--", color="k", alpha=0.6)
plt.xlabel("Time relative to spike [s]")
plt.ylabel("||CoM - baseline|| [m]")
plt.title("Event-aligned CoM deviation norm")
plt.grid(True)
plt.tight_layout()
plt.show()


# 3) Overlay event-aligned ||v||^2
plt.figure(figsize=(11, 5))
for k, r in enumerate(results):
    plt.plot(r["window_t"], r["window_vel2"], alpha=0.7, label=f"Spike {k+1}")
plt.axvline(0.0, linestyle="--", color="k", alpha=0.6)
plt.xlabel("Time relative to spike [s]")
plt.ylabel("||v||^2")
plt.title("Event-aligned velocity norm squared")
plt.grid(True)
plt.tight_layout()
plt.show()