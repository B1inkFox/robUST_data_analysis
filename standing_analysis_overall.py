import numpy as np
import matplotlib.pyplot as plt

from lib.extract_trial_data import *
from lib.detect_spikes import *
from lib.standing_lib import *
from lib.data_lookup import *
from lib.stat_lib import *

DT = 0.01
BASELINE_WINDOW = [-200, -100]
INTEGRAL_WINDOW = [0, 400]

subject_files = [
    (S1_STAND_FILE, S1_STAND_CNT),
    (S2_STAND_FILE, S2_STAND_CNT),
    (S3_STAND_FILE, S3_STAND_CNT),
    (S4_STAND_FILE, S4_STAND_CNT),
    (S5_STAND_FILE, S5_STAND_CNT),
    (S6_STAND_FILE, S6_STAND_CNT),
]

trial_labels = ["Transparent", "Impedance", "MPC"]

all_subject_summary_com = []
all_subject_summary_vel = []

all_subject_results_com = []
all_subject_results_vel = []


for subject_idx, (csv_path_list, num_spikes_list) in enumerate(subject_files):

    subject_summary_com = []
    subject_summary_vel = []

    subject_results_com = []
    subject_results_vel = []

    for i in range(len(csv_path_list)):

        csv_path = csv_path_list[i]
        num_spikes = num_spikes_list[i]

        timestamps, CoM, Ee, Fp1, Fp2 = extract_kinematics_and_forceplates(csv_path)
        timestamps, goal_force, goal_torque, goal_position, goal_euler, goal_velocity, goal_angular_velocity = extract_goals(csv_path)

        t = np.array(timestamps)
        com = np.array([T[0:3, 3] for T in CoM]) - np.array(goal_position)

        impulses, com_spikes, vel_spikes = get_impulses_and_spikes(
            t, com, num_spikes, plot=False
        )

        results_com, summary_com = analyze_position_data(
            com,
            impulses,
            com_spikes,
            BASELINE_WINDOW,
            INTEGRAL_WINDOW,
            plot=False,
            settle_time_threshold=1 / np.e
        )

        results_vel, summary_vel = analyze_velocity_data(
            com,
            impulses,
            vel_spikes,
            INTEGRAL_WINDOW,
            plot=False
        )

        subject_summary_com.append(summary_com)
        subject_summary_vel.append(summary_vel)

        subject_results_com.append(results_com)
        subject_results_vel.append(results_vel)

    all_subject_summary_com.append(subject_summary_com)
    all_subject_summary_vel.append(subject_summary_vel)

    all_subject_results_com.append(subject_results_com)
    all_subject_results_vel.append(subject_results_vel)


# Position metrics
plot_subject_level_across_conditions(
    all_subject_summary_com,
    "peak_error",
    trial_labels,
    ylabel="Peak CoM deviation [m]",
    title="Peak CoM Positional Deviation"
)

plot_subject_level_across_conditions(
    all_subject_summary_com,
    "settling_time_samples",
    trial_labels,
    ylabel="Settling time [s]",
    title="CoM Position Settling Time (Threshold=1/e)"
)

plot_subject_level_across_conditions(
    all_subject_summary_com,
    "integrated_error",
    trial_labels,
    ylabel="Integrated CoM deviation",
    title="Integrated CoM Positional Deviation"
)

# Velocity metrics
plot_subject_level_across_conditions(
    all_subject_summary_vel,
    "peak_speed",
    trial_labels,
    ylabel="Peak speed [m/s]",
    title="Peak CoM Speed"
)

plot_subject_level_across_conditions(
    all_subject_summary_vel,
    "integrated_kinetic_energy",
    trial_labels,
    ylabel="||v||^2",
    title="Integrated kinetic-energy proxy"
)

# Position metrics
analyze_metric(all_subject_summary_com, "peak_error", trial_labels)
analyze_metric(all_subject_summary_com, "settling_time_samples", trial_labels)
analyze_metric(all_subject_summary_com, "integrated_error", trial_labels)

# Velocity metrics
analyze_metric(all_subject_summary_vel, "peak_speed", trial_labels)
analyze_metric(all_subject_summary_vel, "integrated_kinetic_energy", trial_labels)
