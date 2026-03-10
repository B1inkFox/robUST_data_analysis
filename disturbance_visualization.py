import numpy as np
import matplotlib.pyplot as plt

from lib.extract_trial_data import extract_kinematics_and_forceplates, extract_goals
from lib.kernels import *
from lib.detect_spikes import detect_known_number_of_spikes, auto_prominence

csv_path_tran = "Darren_grfmpc_experiment/Darren_Standing_Tran_1_2026-02-24_11-27-20.csv" 
csv_path_imp = "Darren_grfmpc_experiment/Darren_Standing_Imp_1_2026-02-24_12-12-11.csv"
csv_path_mpc = "Darren_grfmpc_experiment/Darren_Standing_MPC_1_2026-02-24_11-29-49.csv"

timestamps, CoM, EE, FP1, FP2 = extract_kinematics_and_forceplates(csv_path_tran)
timestamps, goal_force, goal_torque, goal_position, goal_euler, goal_velocity, goal_angular_velocity = extract_goals(csv_path_tran)

t = np.array(timestamps)
com = np.array([T[0:3, 3] for T in CoM]) - np.array(goal_position)

gauss_kernel = gaussian_kernel(sigma=1, radius=3)  
first_dir_kernel = first_derivative_kernel()
second_dir_kernel = second_derivative_kernel()

com_magnitude = np.sqrt(np.sum(np.square(com), axis=1))
com_magnitude_smoothed = np.convolve(com_magnitude, gauss_kernel, mode='same')
com_velocity = np.convolve(com_magnitude, first_dir_kernel, mode='same')
com_velocity_smoothed = np.convolve(com_velocity, gaussian_kernel(sigma=2, radius=6), mode='same')
com_acceleration = np.convolve(com_magnitude, second_dir_kernel, mode='same')

peak_idx, peak_times, peak_vals, props = detect_known_number_of_spikes(
    signal=com_magnitude_smoothed,
    timestamps=t,
    num_spikes=10,
    min_prominence=auto_prominence(com_magnitude_smoothed, frac=0.25),
    min_width=20,       # in samples; tune this
    min_distance=20,    # in samples; tune this
)
print("Peak times:", peak_times)

plt.figure(figsize=(10, 5))
plt.plot(t, com_magnitude, label="COM Position (Smoothed)")
#plt.plot(t, com_velocity, label="COM Velocity (Smoothed)")
#plt.plot(t, com_acceleration, label="COM Acc (Smoothed)")
for idx in peak_idx:
    plt.axvline(t[idx], linestyle="--", alpha=0.4, color="red")

for idx in peak_idx:
    plt.axvline(t[idx+300], linestyle="--", alpha=0.4, color="blue")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
