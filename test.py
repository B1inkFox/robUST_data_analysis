import numpy as np
import matplotlib.pyplot as plt

from lib.extract_trial_data import extract_kinematics_and_forceplates, extract_goals
from lib.kernels import *
from scipy.signal import find_peaks


#file_path = "robust_data/Darren_grfmpc_experiment/Darren_SEBT_Tran_1_2026-02-24_12-33-32.csv"
#file_path = "robust_data/Darren_grfmpc_experiment/Darren_SEBT_Imp_1_2026-02-24_12-41-56.csv"
file_path = "robust_data/Darren_grfmpc_experiment/Darren_SEBT_MPC_1_2026-02-24_12-37-15.csv"

timestamps, CoM, Ee, Fp1, Fp2 = extract_kinematics_and_forceplates(file_path)
timestamp, goal_force, goal_torque, goal_position, goal_euler, goal_velocity, goal_angular_velocity = extract_goals(file_path)

t = np.array(timestamps)
ee_pos = np.array([T[0:3, 3] for T in Ee])
R = np.array([T[0:3, 0:3] for T in Ee])

# Clamp for numerical safety
r20 = np.clip(R[:, 2, 0], -1.0, 1.0)
# pitch about Y
pitch = np.arcsin(-r20)
# roll about X
roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])

gk = gaussian_kernel(sigma=1, radius=3)
ee_pos_smooth = smooth_3d(ee_pos, gk)
gk_rot = gaussian_kernel(sigma=3, radius=10)
pitch_smooth = smooth_1d(pitch, gk_rot)
roll_smooth = smooth_1d(roll, gk_rot)

ee_x_pos = np.maximum(ee_pos_smooth[:, 0], 0.0)
ee_y_pos = np.maximum(ee_pos_smooth[:, 1], 0.0)

def auto_prominence(signal, frac=0.2):
    """
    A rough automatic prominence scale.
    Uses a fraction of the signal's robust range.
    """
    signal = np.asarray(signal)
    q10, q90 = np.percentile(signal, [10, 90])
    return frac * (q90 - q10)



plt.figure(figsize=(10, 5))
#plt.plot(t, ee_x_pos, label="EE +x")
#plt.plot(t, ee_y_pos, label="EE +y")
plt.plot(t, pitch_smooth, label="Pitch (rad)")
plt.plot(t, roll_smooth, label="Roll (rad)")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Smoothed EE Positive X and Y Components")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()