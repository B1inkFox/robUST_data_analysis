import numpy as np
import matplotlib.pyplot as plt

from lib.extract_trial_data import extract_kinematics_and_forceplates
from lib.smoothing import gaussian_kernel
from lib.detect_spikes import detect_known_number_of_spikes, auto_prominence

#csv_path = "Darren_grfmpc_experiment/Darren_Standing_Tran_1_2026-02-24_11-27-20.csv" 
#csv_path = "Darren_grfmpc_experiment/Darren_Standing_Imp_1_2026-02-24_12-12-11.csv"
csv_path = "Darren_grfmpc_experiment/Darren_Standing_MPC_1_2026-02-24_11-29-49.csv"

timestamps, CoM, EE, FP1, FP2 = extract_kinematics_and_forceplates(csv_path)

t = np.array(timestamps)
com_x = np.array([T[0, 3] for T in CoM])
com_y = np.array([T[1, 3] for T in CoM])
com_z = np.array([T[2, 3] for T in CoM])

com = np.sqrt(com_x**2 + com_y**2 + com_z**2)
gaussian_kernel = gaussian_kernel(sigma=1, radius=3)  
com_smoothed = np.convolve(com, gaussian_kernel, mode='same')

peak_idx, peak_times, peak_vals, props = detect_known_number_of_spikes(
    signal=com_smoothed,
    timestamps=t,
    num_spikes=10,
    min_prominence=auto_prominence(com_smoothed, frac=0.25),
    min_width=20,       # in samples; tune this
    min_distance=20,    # in samples; tune this
)
print("Peak times:", peak_times)

plt.figure(figsize=(10, 5))
plt.plot(t, com_smoothed, label="COM Magnitude (Smoothed)")
for idx in peak_idx:
    plt.axvline(t[idx], linestyle="--", alpha=0.4, color="red")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
