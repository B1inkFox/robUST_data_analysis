import numpy as np
from lib.extract_trial_data import extract_kinematics_and_forceplates, extract_goals
from lib.kernels import gaussian_kernel, smooth_3d
from lib.detect_spikes import detect_known_number_of_spikes, auto_prominence    
import matplotlib.pyplot as plt


file_path = "robust_data/Darren_grfmpc_experiment/Darren_Standing_Tran_1_2026-02-24_11-27-20.csv"   # <-- change this to your file
file_path = "robust_data/Darren_grfmpc_experiment/Darren_Standing_Imp_1_2026-02-24_12-12-11.csv"
file_path = "robust_data/Darren_grfmpc_experiment/Darren_Standing_MPC_1_2026-02-24_11-29-49.csv"

# ---- 2. Call Function 1 ----
timestamps, CoM, EE, FP1, FP2 = extract_kinematics_and_forceplates(file_path)
timestamp, goal_force, goal_torque, goal_position, goal_euler, goal_velocity, goal_angular_velocity = extract_goals(file_path)

t = np.array(timestamps)
dt = 0.01
com = np.array([T[0:3, 3] for T in CoM])
com_velocity = np.gradient(com, dt, axis=0)   # shape (N,3)
com_acceleration = np.gradient(com_velocity, dt, axis=0)   # shape (N,3)
gk = gaussian_kernel(sigma=1, radius=10)
com_smoothed = smooth_3d(com, gk)
com2 = np.sum(com_smoothed**2, axis=1)                 # ||CoM||^2
com_vel_smoothed = smooth_3d(com_velocity, gk)
vel2 = np.sum(com_vel_smoothed**2, axis=1)     
# Detect spikes
peak_idx, peak_times, peak_vals, props = detect_known_number_of_spikes(
    signal=vel2,
    timestamps=t,
    num_spikes=10,
    min_prominence=auto_prominence(vel2, frac=0.25),
    min_width=2,
    min_distance=20,
)


plt.figure(figsize=(11, 5))
plt.plot(t, vel2, label="Smoothed Vel magnitude")
plt.plot(t, com2, label="Smoothed CoM magnitude")
for idx in peak_idx:
    plt.axvline(t[idx-25], linestyle="--", alpha=0.4, color="red")
plt.xlabel("Time [s]")
plt.ylabel("||CoM - goal|| [m]")
plt.title("Detected disturbance times")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()