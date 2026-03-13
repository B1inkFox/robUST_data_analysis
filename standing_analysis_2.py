import numpy as np
import matplotlib.pyplot as plt

from lib.extract_trial_data import *
from lib.detect_spikes import *
from lib.task1_analysis import *

csv_path_tran = "Darren_grfmpc_experiment/Darren_Standing_Tran_1_2026-02-24_11-27-20.csv"
csv_path_imp  = "Darren_grfmpc_experiment/Darren_Standing_Imp_1_2026-02-24_12-12-11.csv"
csv_path_mpc  = "Darren_grfmpc_experiment/Darren_Standing_MPC_1_2026-02-24_11-29-49.csv"

csv_path = csv_path_imp   # choose which file to analyze

DT = 0.01
BASELINE_WINDOW = [-200, -100]
INTEGRAL_WINDOW = [0, 200]
NUM_SPIKES = 11

timestamps, CoM, EE, FP1, FP2 = extract_kinematics_and_forceplates(csv_path)
timestamps, goal_force, goal_torque, goal_position, goal_euler, goal_velocity, goal_angular_velocity = extract_goals(csv_path)

t = np.array(timestamps)
com = np.array([T[0:3, 3] for T in CoM]) - np.array(goal_position)

# impulses is the instant at which perturbation is delivered. It is the left corner of the COM error peak.
# COM spikes is the instant at which COM error peaks, i.e. maximal COM extrusion. Vel spikes is the instant at which COM velocity peaks, which should be close to the impulse time but slightly delayed.
impulses, com_spikes, vel_spikes = get_impulses_and_spikes(t, com, NUM_SPIKES, plot=True)
results_com, summary_com = analyze_position_data(com, impulses, com_spikes, BASELINE_WINDOW, INTEGRAL_WINDOW)
results_vel, summary_vel = analyze_velocity_data(com, impulses, vel_spikes, INTEGRAL_WINDOW)