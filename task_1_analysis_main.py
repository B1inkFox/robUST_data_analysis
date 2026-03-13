import numpy as np
import matplotlib.pyplot as plt

from lib.extract_trial_data import *
from lib.detect_spikes import *
from lib.task1_lib import *

DT = 0.01
BASELINE_WINDOW = [-200, -100]
INTEGRAL_WINDOW = [0, 250]

csv_path_tran = "Darren_grfmpc_experiment/Darren_Standing_Tran_1_2026-02-24_11-27-20.csv"
csv_path_imp  = "Darren_grfmpc_experiment/Darren_Standing_Imp_1_2026-02-24_12-12-11.csv"
csv_path_mpc  = "Darren_grfmpc_experiment/Darren_Standing_MPC_1_2026-02-24_11-29-49.csv"

csv_path_list = [csv_path_tran, csv_path_imp, csv_path_mpc]
num_spikes_list = [10, 11, 10]
trial_labels = ["tran", "imp", "mpc"]

all_summary_com = []
all_summary_vel = []

print_summary = False
for i in range(len(csv_path_list)):

    csv_path = csv_path_list[i]
    num_spikes = num_spikes_list[i]

    timestamps, CoM, EE, FP1, FP2 = extract_kinematics_and_forceplates(csv_path)
    timestamps, goal_force, goal_torque, goal_position, goal_euler, goal_velocity, goal_angular_velocity = extract_goals(csv_path)

    t = np.array(timestamps)
    com = np.array([T[0:3, 3] for T in CoM]) - np.array(goal_position)

    # impulses is the instant at which perturbation is delivered. It is the left corner of the COM error peak.
    # COM spikes is the instant at which COM error peaks, i.e. maximal COM extrusion. Vel spikes is the instant at which COM velocity peaks, which should be close to the impulse time but slightly delayed.
    impulses, com_spikes, vel_spikes = get_impulses_and_spikes(t, com, num_spikes, plot=True)
    results_com, summary_com = analyze_position_data(com, impulses, com_spikes, BASELINE_WINDOW, INTEGRAL_WINDOW, plot=True)
    results_vel, summary_vel = analyze_velocity_data(com, impulses, vel_spikes, INTEGRAL_WINDOW, plot=True)
    
    all_summary_com.append(summary_com)
    all_summary_vel.append(summary_vel)

    if print_summary:
        print("\n=================================================")
        print(f"Experiment: {csv_path}")
        print("=================================================")

        print_summary("Position Metrics", summary_com)
        print_summary("Velocity Metrics", summary_vel)

# Position metrics
plot_metric_summary(
    "peak_error",
    all_summary_com,
    trial_labels,
    ylabel="Peak CoM deviation [m]"
)

plot_metric_summary(
    "settling_time_samples",
    all_summary_com,
    trial_labels,
    ylabel="Settling time [s]"
)

plot_metric_summary(
    "integrated_error",
    all_summary_com,
    trial_labels,
    ylabel="Integrated CoM deviation"
)

# Velocity metrics
plot_metric_summary(
    "peak_speed",
    all_summary_vel,
    trial_labels,
    ylabel="Peak speed [m/s]"
)

plot_metric_summary(
    "integrated_kinetic_energy",
    all_summary_vel,
    trial_labels,
    ylabel="Integrated kinetic-energy proxy"
)