# extract_trial_data.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd


# Fixed layout (0-indexed, end-exclusive)
IDX_TIMESTAMP = 0

IDX_COM = (1, 17)      # 16
IDX_EE  = (17, 33)     # 16

IDX_FP1 = (33, 39)     # 6
IDX_FP2 = (39, 45)     # 6

IDX_GOAL_F = (45, 48)  # 3
IDX_GOAL_T = (48, 51)  # 3
IDX_GOAL_P = (51, 54)  # 3
IDX_GOAL_E = (54, 57)  # 3
IDX_GOAL_V = (57, 60)  # 3
IDX_GOAL_W = (60, 63)  # 3

N_COLS_EXPECTED = 63

def _read_numeric_matrix(csv_path: str | Path) -> np.ndarray:
    """
    Reads CSV (comma-separated), skips the header row, returns numeric array (N, D).
    No header-based indexing is used.
    """
    csv_path = Path(csv_path)

    # Read everything but header as raw numeric matrix
    df = pd.read_csv(csv_path, sep=",", header=0, index_col=False)
    data = df.to_numpy(dtype=float, copy=False)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    if data.shape[1] != N_COLS_EXPECTED:
        raise ValueError(
            f"Expected at least {N_COLS_EXPECTED} columns, got {data.shape[1]}.\n"
        )

    return data


def extract_kinematics_and_forceplates(
    csv_path: str | Path,
    sanity_check_homogeneous: bool = True,
) -> Tuple[
    List[float],      # timestamp
    List[np.ndarray], # CoM 4x4
    List[np.ndarray], # EE  4x4
    List[np.ndarray], # FP1 (6,)
    List[np.ndarray], # FP2 (6,)
]:
    """
    Returns timestamp, CoM, EE, FP1, FP2 using fixed column positions.
    """
    X = _read_numeric_matrix(csv_path)
    #print(X.shape)
    #print(X[0])

    ts = X[:, IDX_TIMESTAMP].astype(float).tolist()

    com_flat = X[:, IDX_COM[0]:IDX_COM[1]]  # (N,16)
    ee_flat  = X[:, IDX_EE[0]:IDX_EE[1]]    # (N,16)

    fp1 = X[:, IDX_FP1[0]:IDX_FP1[1]]       # (N,6)
    fp2 = X[:, IDX_FP2[0]:IDX_FP2[1]]       # (N,6)

    CoM = [row.reshape(4, 4) for row in com_flat]
    EE  = [row.reshape(4, 4) for row in ee_flat]
    FP1 = [row.copy() for row in fp1]
    FP2 = [row.copy() for row in fp2]

    return ts, CoM, EE, FP1, FP2


def extract_goals(
    csv_path: str | Path,
) -> Tuple[
    List[float],      # timestamp
    List[np.ndarray], # Goal Force (3,)
    List[np.ndarray], # Goal Torque (3,)
    List[np.ndarray], # Goal Position (3,)
    List[np.ndarray], # Goal Euler (3,)
    List[np.ndarray], # Goal Velocity (3,)
    List[np.ndarray], # Goal Angular Velocity (3,)
]:
    """
    Function 2 (hard-coded):
    Returns timestamp + goal signals using fixed column positions.
    """
    X = _read_numeric_matrix(csv_path)

    ts = X[:, IDX_TIMESTAMP].astype(float).tolist()

    GF = [row.copy() for row in X[:, IDX_GOAL_F[0]:IDX_GOAL_F[1]]]
    GT = [row.copy() for row in X[:, IDX_GOAL_T[0]:IDX_GOAL_T[1]]]
    GP = [row.copy() for row in X[:, IDX_GOAL_P[0]:IDX_GOAL_P[1]]]
    GE = [row.copy() for row in X[:, IDX_GOAL_E[0]:IDX_GOAL_E[1]]]
    GV = [row.copy() for row in X[:, IDX_GOAL_V[0]:IDX_GOAL_V[1]]]
    GW = [row.copy() for row in X[:, IDX_GOAL_W[0]:IDX_GOAL_W[1]]]

    return ts, GF, GT, GP, GE, GV, GW