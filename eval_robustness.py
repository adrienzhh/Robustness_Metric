"""
Robustness Metrics Evaluation 

Evaluates the robustness of trajectory estimates by computing:
- RPE (Relative Pose Error) for translation and rotation
- F-scores at configurable thresholds
- AUC (Area Under Curve) metrics across threshold ranges

Supports both csv and txt (TUM formats) 
"""

import sys
import os
import argparse
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'script'))
sys.path.append(os.path.join(script_dir, 'python_wrapper'))

from evo.core import metrics, trajectory
from evo.core.units import Unit
from evo.tools import log, plot, file_interface
from evo.tools.settings import SETTINGS
from evo.core import sync
from robustness import RobustnessMetric
from pyhocon import ConfigFactory
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation
from evo.core.metrics import PoseRelation
import evo.main_rpe as main_rpe
import numpy as np

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
import numpy as np

print(f"Current working directory: {os.getcwd()}")

def read_trajectory_file(filepath):
    """
    Read trajectory file supporting both CSV and TUM formats.
    Automatically converts nanosecond timestamps to seconds.

    CSV format expected: timestamp, x, y, z, qx, qy, qz, qw
    TUM format: timestamp x y z qx qy qz qw
    """
    file_ext = os.path.splitext(filepath)[1].lower()

    if file_ext == '.csv':
        # Read CSV file
        df = pd.read_csv(filepath)

        # Handle different possible column names
        possible_timestamp_cols = ['timestamp', 'time', 't', 'Timestamp', 'Time']
        timestamp_col = None
        for col in possible_timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            # Assume first column is timestamp
            timestamp_col = df.columns[0]

        timestamps = df[timestamp_col].values

        # Convert nanoseconds to seconds if needed (nanoseconds are typically > 1e10)
        if timestamps[0] > 1e10:
            print(f"Converting timestamps from nanoseconds to seconds for {filepath}")
            timestamps = timestamps / 1e9

        # Try to extract position and orientation columns
        # Common formats: x,y,z,qx,qy,qz,qw or tx,ty,tz,qx,qy,qz,qw
        position_cols = []
        orientation_cols = []

        # Look for position columns
        for x_col in ['x', 'tx', 'X', 'pos_x']:
            if x_col in df.columns:
                y_col = x_col.replace('x', 'y').replace('X', 'Y')
                z_col = x_col.replace('x', 'z').replace('X', 'Z')
                if y_col in df.columns and z_col in df.columns:
                    position_cols = [x_col, y_col, z_col]
                    break

        # If not found, try using column indices
        if not position_cols:
            if len(df.columns) >= 8:
                position_cols = [df.columns[1], df.columns[2], df.columns[3]]

        # Look for orientation columns (quaternion: qx, qy, qz, qw)
        for qx_col in ['qx', 'QX', 'q_x', 'ori_x']:
            if qx_col in df.columns:
                qy_col = qx_col.replace('x', 'y').replace('X', 'Y')
                qz_col = qx_col.replace('x', 'z').replace('X', 'Z')
                qw_col = qx_col.replace('x', 'w').replace('X', 'W')
                if qy_col in df.columns and qz_col in df.columns and qw_col in df.columns:
                    orientation_cols = [qx_col, qy_col, qz_col, qw_col]
                    break

        # If not found, try using column indices
        if not orientation_cols:
            if len(df.columns) >= 8:
                orientation_cols = [df.columns[4], df.columns[5], df.columns[6], df.columns[7]]

        positions = df[position_cols].values
        orientations = df[orientation_cols].values

        # CSV quaternions might be in [x, y, z, w] or [w, x, y, z] format
        # We need [w, x, y, z] format for evo's PoseTrajectory3D
        # Check if first quaternion's last component looks like w (close to ±1 or normalized)
        sample_quat = orientations[0]
        quat_norm = np.linalg.norm(sample_quat)

        # If the quaternion is in [x, y, z, w] format (scipy convention), reorder to [w, x, y, z]
        # Heuristic: if last element has larger absolute value, it's likely the w component
        if abs(sample_quat[3]) > abs(sample_quat[0]):
            # Assume [x, y, z, w] format, convert to [w, x, y, z]
            orientations = np.roll(orientations, 1, axis=1)

        # Create PoseTrajectory3D trajectory (positions, quaternions_wxyz, timestamps)
        return PoseTrajectory3D(positions, orientations, timestamps)
    else:
        # Use existing TUM reader for .txt or other formats
        traj = file_interface.read_tum_trajectory_file(filepath)

        # Check if timestamps are in nanoseconds and convert to seconds
        if len(traj.timestamps) > 0 and traj.timestamps[0] > 1e10:
            print(f"Converting timestamps from nanoseconds to seconds for {filepath}")
            timestamps = np.array(traj.timestamps) / 1e9
            # evo uses [w, x, y, z] format for quaternions
            # PoseTrajectory3D(positions, quaternions_wxyz, timestamps)
            return PoseTrajectory3D(
                traj.positions_xyz,
                traj.orientations_quat_wxyz,
                timestamps
            )

        return traj

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    return ConfigFactory.parse_file(config_path)

def format_results(auc_result):
    result = "-------------\n"
    result += "Thresholds | F-score (Trans) | F-score (Rot)\n"
    result += "-----------+------------------+--------------\n"
    for t, ft, fr in zip(auc_result['thresholds'], 
                         auc_result['fscore_transes'], 
                         auc_result['fscore_rots']):
        result += f"{t:9.3f} | {ft:16.4f} | {fr:12.4f}\n"
    result += "-----------+------------------+--------------\n"

    fscore_area_trans = auc_result['fscore_area_trans'].item() if hasattr(auc_result['fscore_area_trans'], 'item') else auc_result['fscore_area_trans']
    fscore_area_rot = auc_result['fscore_area_rot'].item() if hasattr(auc_result['fscore_area_rot'], 'item') else auc_result['fscore_area_rot']

    result += f"AUC (Trans): {fscore_area_trans:.4f}\n"
    result += f"AUC (Rot)  : {fscore_area_rot:.4f}\n"
    return result

def process_trajectory_pair(ref_file, est_file, config):
    max_diff = 0.2
    trans_threshold = 0.1
    rot_threshold = 0.1

    threshold_start = config.get_float('parameters.threshold_start')
    threshold_end = config.get_float('parameters.threshold_end')
    threshold_interval = threshold_start 


    traj_ref = read_trajectory_file(ref_file)
    full_len = len(traj_ref.positions_xyz)
    print("Matched frames count:", full_len)

    traj_est = read_trajectory_file(est_file)

    offset_2 = abs(traj_est.timestamps[0] - traj_ref.timestamps[0])
    offset_2 *= -1 if traj_est.timestamps[0] > traj_ref.timestamps[0] else 1

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff, offset_2)

    print("Est Size:", len(traj_est.positions_xyz))

    result_trans = main_rpe.rpe(
        traj_ref, traj_est,
        est_name='RPE translation',
        pose_relation=PoseRelation.translation_part,
        delta=1.0, delta_unit=Unit.frames,
        all_pairs=False, align=True, correct_scale=False,
        support_loop=False
    )
    rpe_trans_result = result_trans.np_arrays["error_array"]

    rpe_rot = metrics.RPE(
        PoseRelation.rotation_angle_deg,
        delta=1.0, delta_unit=Unit.frames,
        all_pairs=False
    )
    rpe_rot.process_data((traj_ref, traj_est))
    rpe_rot_result = np.radians(rpe_rot.error)

    fscore_trans, fscore_rot = RobustnessMetric.calc_fscore(
        rpe_trans_result, rpe_rot_result,
        full_len, trans_threshold, rot_threshold
    )

    auc_result = RobustnessMetric.eval_robustness_batch(
        rpe_trans_result, rpe_rot_result,
        full_len, threshold_start, threshold_end, threshold_interval
    )

    return fscore_trans, fscore_rot, auc_result


def main(config_path, plot_mode):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Using config file: {config_path}")
    
    config = load_config(config_path)
    trajectory_pairs = config.get_list('trajectory_pairs')
    
    for i, pair in enumerate(trajectory_pairs):
        ref_file = pair.get_string('reference')
        est_file = pair.get_string('estimated')

        print(f"Reference: {ref_file}")
        print(f"Estimated: {est_file}")
        
        fscore_trans, fscore_rot, auc_result = process_trajectory_pair(ref_file, est_file, config)
        formatted_results = format_results(auc_result)
        
        print("---------------------------------")
        print("\nAUC Result:")
        print(formatted_results)
        
        RobustnessMetric.save_results(ref_file, est_file, fscore_trans, fscore_rot, auc_result)
        
        if plot_mode:
            RobustnessMetric.plot_robustness_metrics(auc_result, ref_file, est_file)

    print("\nScript execution completed. Results have been saved in the reference directories.")
    if plot_mode:
        print("Plots have been generated for each trajectory pair.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory Robustness Analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of robustness metrics")
    args = parser.parse_args()
    main(config_path=args.config, plot_mode=args.plot)