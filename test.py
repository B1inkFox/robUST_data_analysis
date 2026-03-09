import numpy as np
from lib.extract_trial_data import extract_kinematics_and_forceplates


file_path = "Darren_grfmpc_experiment/Darren_Standing_Tran_1_2026-02-24_11-27-20.csv"   # <-- change this to your file

# ---- 2. Call Function 1 ----
timestamps, CoM, EE, FP1, FP2 = extract_kinematics_and_forceplates(file_path)

# ---- 3. Basic inspection ----
print(f"Total samples: {len(timestamps)}")
print(f"First timestamp: {timestamps[0]:.4f} s")

print("\nFirst CoM homogeneous matrix:")
print(CoM[0])

print("\nFirst EE homogeneous matrix:")
print(EE[0])

print("\nFirst FP1 measurement [Fx, Fy, Fz, CoPx, CoPy, CoPz]:")
print(FP1[0])

print("\nFirst FP2 measurement [Fx, Fy, Fz, CoPx, CoPy, CoPz]:")
print(FP2[0])

# ---- 4. Example: compute CoM position trajectory ----
# Extract translation part from CoM matrices
com_positions = np.array([T[0:3, 3] for T in CoM])

print("\nFirst 5 CoM positions:")
print(com_positions[:5])
