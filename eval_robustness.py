import sys
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for headless environments

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'script'))
sys.path.append(os.path.join(script_dir, 'python_wrapper'))

from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = 'Agg'
from evo.core import metrics, trajectory
from evo.core.units import Unit
from evo.tools import log, plot, file_interface
from robustness import RobustnessMetric
from pyhocon import ConfigFactory
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation
from evo.core.metrics import PoseRelation
import evo.main_rpe as main_rpe
import numpy as np
import math

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

# def resample_ground_truth(traj_ref, target_timestamps):

#     t_ref = np.array(traj_ref.timestamps)
#     pts   = np.array(traj_ref.positions_xyz)
#     fx = interp1d(t_ref, pts[:,0], kind='linear', fill_value='extrapolate')
#     fy = interp1d(t_ref, pts[:,1], kind='linear', fill_value='extrapolate')
#     fz = interp1d(t_ref, pts[:,2], kind='linear', fill_value='extrapolate')
#     t_tgt = np.array(target_timestamps)
#     new_pts = np.vstack((fx(t_tgt), fy(t_tgt), fz(t_tgt))).T


#     # evo uses [w, x, y, z] format, scipy uses [x, y, z, w]
#     quats_wxyz = np.array(traj_ref.orientations_quat_wxyz)
#     # Convert from [w, x, y, z] to [x, y, z, w] for scipy
#     quats_xyzw = np.roll(quats_wxyz, -1, axis=1)
#     rot_seq = Rotation.from_quat(quats_xyzw)
#     slerp   = Slerp(t_ref, rot_seq)
#     new_rots = slerp(t_tgt)
#     # Convert back from [x, y, z, w] to [w, x, y, z] for evo
#     new_quats_xyzw = new_rots.as_quat()
#     new_quats_wxyz = np.roll(new_quats_xyzw, 1, axis=1)

#     # PoseTrajectory3D(positions, quaternions_wxyz, timestamps)
#     return PoseTrajectory3D(new_pts, new_quats_wxyz, t_tgt)

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

def align_estimator_to_gt(traj_ref, traj_est):
    """
    Align the estimated trajectory to GT timestamps by reusing the closest pose.
    This allows low-frequency estimators to be evaluated fairly while still
    penalizing large gaps via translation/rotation error.
    """
    t_ref = np.array(traj_ref.timestamps)
    t_est = np.array(traj_est.timestamps)

    if len(t_ref) == 0 or len(t_est) == 0:
        return None, None, np.array([], dtype=bool)

    min_est = t_est[0]
    max_est = t_est[-1]

    valid_mask = (t_ref >= min_est) & (t_ref <= max_est)
    if not np.any(valid_mask):
        return None, None, valid_mask

    t_overlap = t_ref[valid_mask]

    candidate_idx = np.searchsorted(t_est, t_overlap, side='left')
    candidate_idx = np.clip(candidate_idx, 0, len(t_est) - 1)
    prev_idx = np.clip(candidate_idx - 1, 0, len(t_est) - 1)

    use_prev = (
        (candidate_idx > 0)
        & (
            np.abs(t_est[prev_idx] - t_overlap)
            < np.abs(t_est[candidate_idx] - t_overlap)
        )
    )
    candidate_idx[use_prev] = prev_idx[use_prev]

    ref_positions = np.array(traj_ref.positions_xyz)[valid_mask]
    ref_rotations = np.array(traj_ref.orientations_quat_wxyz)[valid_mask]
    matched_positions = np.array(traj_est.positions_xyz)[candidate_idx]
    matched_rotations = np.array(traj_est.orientations_quat_wxyz)[candidate_idx]

    traj_ref_synced = PoseTrajectory3D(ref_positions, ref_rotations, t_overlap)
    traj_est_synced = PoseTrajectory3D(matched_positions, matched_rotations, t_overlap)

    return traj_ref_synced, traj_est_synced, valid_mask

def process_trajectory_pair(ref_file, est_file, config):
    # --- 1. Load Configuration ---
    # Default values are safe safeguards if config is missing keys


    # --- 2. Read Files ---
    traj_ref = read_trajectory_file(ref_file)
    traj_est = read_trajectory_file(est_file)

    # [CRITICAL STEP] 
    # Capture the full length of the Ground Truth *before* synchronization.
    # This represents the full mission duration.
    total_gt_frames = len(traj_ref.positions_xyz)
    print(f"  > GT Frames (Total Mission): {total_gt_frames}")
    print(f"  > Est Frames (Submitted):    {len(traj_est.positions_xyz)}")

    # --- 3. [THE FIX] Interpolate Sparse Estimated Trajectory ---
    # If estimate is sparse, we interpolate it to match the high-frequency Ground Truth.
    # This removes penalties for low update rates while preserving trajectory shape error.
    
    # 3a. Interpolate Position (Linear)
    # Using interp1d with extrapolation might be dangerous if timestamps don't align perfectly at ends,
    # but a later alignment pass will handle the intersection. We interpolate over the whole range first.
    t_est = np.array(traj_est.timestamps)
    pos_est = np.array(traj_est.positions_xyz)
    
    # Check if we need interpolation (if est density is much lower than GT)
    # E.g., if GT has 10x more frames
    if total_gt_frames > 2 * len(traj_est.positions_xyz):
        print("  > Detected sparse trajectory. Interpolating to GT timestamps...")
        
        # We only interpolate at GT timestamps that fall within the Est time range
        # to avoid wild extrapolation.
        t_ref = np.array(traj_ref.timestamps)
        
        # [CRITICAL] Ensure unique timestamps in GT before interpolation
        # Some datasets (or parsing) might produce duplicate timestamps, leading to double counting
        t_ref, unique_indices = np.unique(t_ref, return_index=True)
        # We need to filter positions/orientations too if we were using t_ref for more than just lookup
        # But here we just use t_ref to find target timestamps.
        
        valid_mask = (t_ref >= t_est[0]) & (t_ref <= t_est[-1])
        t_target = t_ref[valid_mask]
        
        # Linear Position Interpolation
        fx = interp1d(t_est, pos_est[:,0], kind='linear')
        fy = interp1d(t_est, pos_est[:,1], kind='linear')
        fz = interp1d(t_est, pos_est[:,2], kind='linear')
        new_pos = np.vstack((fx(t_target), fy(t_target), fz(t_target))).T
        
        # Slerp Rotation Interpolation
        quat_est_wxyz = np.array(traj_est.orientations_quat_wxyz)
        # scipy needs [x, y, z, w]
        quat_est_xyzw = np.roll(quat_est_wxyz, -1, axis=1)
        rot_spline = Slerp(t_est, Rotation.from_quat(quat_est_xyzw))
        new_rot_xyzw = rot_spline(t_target).as_quat()
        # convert back to [w, x, y, z]
        new_rot_wxyz = np.roll(new_rot_xyzw, 1, axis=1)
        
        # Create new dense trajectory object
        traj_est = PoseTrajectory3D(new_pos, new_rot_wxyz, t_target)
        print(f"  > Interpolated Est Frames: {len(traj_est.positions_xyz)}")

    # --- 4. Synchronization via Nearest-Neighbor Reuse ---
    traj_ref_synced, traj_est_synced, _ = align_estimator_to_gt(traj_ref, traj_est)
    
    if traj_ref_synced is None or len(traj_ref_synced.positions_xyz) < 2:
        print("  ! WARNING: Could not find overlapping timestamps. Returning zero scores.")
        return 0.0, 0.0, {
            'thresholds': [], 'fscore_transes': [], 'fscore_rots': [], 
            'fscore_area_trans': 0.0, 'fscore_area_rot': 0.0
        }
    
    matched_count = len(traj_ref_synced.positions_xyz)
    coverage_ratio = matched_count / total_gt_frames if total_gt_frames > 0 else 0.0
    print(f"  > Matched Frames (Reuse allowed): {matched_count}")
    print(f"  > Coverage Ratio: {coverage_ratio*100:.2f}% of GT timeline")

    # --- 5. Calculate RPE (Translation) ---
    # Using main_rpe wrapper ensures consistent unit scaling and alignment
    result_trans = main_rpe.rpe(
        traj_ref_synced, traj_est_synced,
        est_name='RPE translation',
        pose_relation=PoseRelation.translation_part,
        delta=1.0, delta_unit=Unit.frames,
        all_pairs=False, align=True, correct_scale=False,
        support_loop=False
    )
    rpe_trans_errors = result_trans.np_arrays["error_array"]

    # --- 5. Calculate RPE (Rotation) ---
    result_rot = main_rpe.rpe(
        traj_ref_synced, traj_est_synced,
        est_name='RPE rotation',
        pose_relation=PoseRelation.rotation_angle_deg,
        delta=1.0, delta_unit=Unit.frames,
        all_pairs=False, align=True, correct_scale=False,
        support_loop=False
    )
    rpe_rot_errors = result_rot.np_arrays["error_array"]

    # DEBUG: Print RPE statistics
    print(f"  [DEBUG] RPE Translation Stats: Min={np.min(rpe_trans_errors):.4f}, Mean={np.mean(rpe_trans_errors):.4f}, Max={np.max(rpe_trans_errors):.4f}")
    print(f"  [DEBUG] RPE Rotation Stats:    Min={np.min(rpe_rot_errors):.4f}, Mean={np.mean(rpe_rot_errors):.4f}, Max={np.max(rpe_rot_errors):.4f}")

    # --- 6. Coverage Diagnostics (No artificial padding) ---
    gt_duration = traj_ref.timestamps[-1] - traj_ref.timestamps[0]
    est_duration = traj_est.timestamps[-1] - traj_est.timestamps[0]
    print(f"  > Duration GT: {gt_duration:.2f}s, Est: {est_duration:.2f}s")
    
    actual_error_count = len(rpe_trans_errors)
    print(f"  > Actual Error Samples: {actual_error_count}")
    
    rpe_trans_final = rpe_trans_errors
    rpe_rot_final = rpe_rot_errors
    calc_len = len(rpe_trans_final)

    # --- 7. Calculate Metrics ---
    # For single-point results, we pick a "Reasonable" threshold (e.g., x=0.5)
    trans_max = config.get_float('parameters.trans_max_effective_error', 1.0)
    rot_max   = config.get_float('parameters.rot_max_effective_error', 30.0)
    mid_x = 0.5
    trans_mid = trans_max * math.exp(-10.0 * mid_x)
    rot_mid   = rot_max * math.exp(-10.0 * mid_x)

    fscore_trans, fscore_rot = RobustnessMetric.calc_fscore(
        rpe_trans_final, rpe_rot_final,
        total_gt_frames, trans_mid, rot_mid
    )

    auc_result = RobustnessMetric.eval_robustness_batch(
        rpe_trans_final, rpe_rot_final,
        total_gt_frames, config
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
