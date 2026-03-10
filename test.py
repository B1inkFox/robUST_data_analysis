import numpy as np
from lib.extract_trial_data import extract_kinematics_and_forceplates, extract_goals


file_path = "Darren_grfmpc_experiment/Darren_Standing_Tran_1_2026-02-24_11-27-20.csv"   # <-- change this to your file

# ---- 2. Call Function 1 ----
timestamps, CoM, EE, FP1, FP2 = extract_kinematics_and_forceplates(file_path)
timestamp, goal_force, goal_torque, goal_position, goal_euler, goal_velocity, goal_angular_velocity = extract_goals(file_path)

com = np.array([T[0:3, 3] for T in CoM])
goal_com = np.array(goal_position)
print(com.shape)
print(goal_com.shape)

