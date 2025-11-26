"""
Robustness Metric Module

This module provides the RobustnessMetric class for evaluating trajectory estimation robustness.
It implements F-score calculation and Area Under Curve (AUC) computation for assessing the
quality of SLAM/VIO trajectory estimates across multiple error thresholds.

Key Components:
- calc_fscore: Computes F-score from RPE (Relative Pose Error) metrics
- eval_robustness_batch: Evaluates robustness across multiple threshold levels and computes AUC
- save_results: Exports robustness metrics to CSV files
- plot_robustness_metrics: Generates visualization plots showing F-score vs threshold curves

The robustness metric quantifies how well a trajectory estimate performs across different
error tolerance levels, providing a comprehensive assessment beyond single-threshold metrics.
"""

import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from pyhocon import ConfigFactory
from evo.core import sync
from evo.tools import file_interface, plot
from evo.core import metrics
from evo.core.trajectory import PosePath3D
from evo.core.metrics import PoseRelation
from evo.core.result import Result

import numpy as np
from scipy.interpolate import interp1d

class RobustnessMetric:
    def calc_fscore(rpe_trans, rpe_rots, full_len, trans_threshold, rot_threshold):
        """
        Calculate the F-score for RPE and angular RPE 

        Parameters:
        rpe_trans: translational RPE
        rpe_rots: rotational RPE
        full_len: Complete length of GT
        trans_threshold (float)
        rot_threshold (float)
        Returns:
        tuple: F-score for translation and rotational 
        """
        trans_num = 0
        rot_num = 0
        index = 0

        while index < len(rpe_trans) and index < full_len:
            trans_val = rpe_trans[index]
            rot_val = rpe_rots[index]

            if trans_val <= trans_threshold:
                trans_num += 1
            if rot_val <= rot_threshold:
                rot_num += 1

            index += 1

        precision_trans = trans_num / len(rpe_trans)
        precision_rot = rot_num / len(rpe_rots)

        recall_trans = trans_num / full_len
        recall_rot = rot_num / full_len

        fscore_trans = 0.0
        if precision_trans + recall_trans > 0:
            fscore_trans = (2 * precision_trans * recall_trans) / (precision_trans + recall_trans)

        fscore_rot = 0.0
        if precision_rot + recall_rot > 0:
            fscore_rot = (2 * precision_rot * recall_rot) / (precision_rot + recall_rot)

        return fscore_trans, fscore_rot
    
    def eval_robustness_batch(rpe_trans, rpe_rots, full_len, config):
        """
        Evaluate robustness by calculating F-scores across a range of thresholds.
        Correctly handles independent scaling for Translation (meters) and Rotation (degrees).
        """
        
        # --- 1. Load Explicit Thresholds from Config ---
        
        # Translation Limits (Meters)
        t_start = config.get_float('parameters.trans_threshold_start', 0.01) # e.g., 0.01 m
        t_end   = config.get_float('parameters.trans_threshold_end', 1.0)    # e.g., 1.0 m
        
        # Rotation Limits (Degrees)
        r_start = config.get_float('parameters.rot_threshold_start', 0.1)    # e.g., 0.1 deg
        r_end   = config.get_float('parameters.rot_threshold_end', 10.0)     # e.g., 10.0 deg
        
        # Step size for the sweep (applied to Translation)
        # Smaller interval = smoother AUC curve
        t_interval = config.get_float('parameters.threshold_interval', 0.01)

        fscore_area_trans = 0.0
        fscore_area_rot = 0.0
        fscore_transes = []
        fscore_rots = []
        thresholds = []
        
        # We iterate based on the Translation scale (since the paper uses exp(-10*T_trans))
        current_t = t_start
        
        while current_t <= t_end:
            # --- 2. Calculate Current Thresholds ---
            
            # Translation is direct
            T_trans = current_t
            
            # Rotation is Interpolated
            # Calculate progress ratio (0.0 to 1.0)
            if t_end != t_start:
                progress = (current_t - t_start) / (t_end - t_start)
            else:
                progress = 1.0
                
            # Map progress to Rotation Range
            T_rot_deg = r_start + progress * (r_end - r_start)
            
            # --- 3. Calculate F-Score ---
            fscore_trans, fscore_rot = RobustnessMetric.calc_fscore(
                rpe_trans, rpe_rots, full_len, T_trans, T_rot_deg
            )
            
            # --- 4. Calculate Weighting (Paper Formula) ---
            # The paper uses x = exp(-10 * T) relative to the translation metric.
            # However, this weighting is extremely aggressive for large error regimes (e.g., Rotation).
            # We relax this to exp(-2 * T) so that larger errors still contribute to the AUC.
            
            # Using -2.0 allows significant contribution even at T=0.5 (exp(-1) ~= 0.36)
            # whereas -10.0 gave negligible contribution at T=0.5 (exp(-5) ~= 0.006)
            weight_factor = -10.0 
            
            val_minus = math.exp(weight_factor * (current_t - t_interval * 0.5))
            val_plus  = math.exp(weight_factor * (current_t + t_interval * 0.5))
            x_axis_len = val_minus - val_plus
            
            if (current_t - t_interval * 0.5) < 0.0:
                x_axis_len = 1.0 - val_plus 
            
            if x_axis_len < 0:
                x_axis_len = 0 # Prevent negative area if step is weird
                
            # --- 5. Integrate ---
            fscore_area_trans += fscore_trans * x_axis_len
            fscore_area_rot   += fscore_rot   * x_axis_len
            
            fscore_transes.append(fscore_trans)
            fscore_rots.append(fscore_rot)
            thresholds.append(current_t)
            
            current_t += t_interval

        return {
            'fscore_transes': fscore_transes,
            'fscore_rots': fscore_rots,
            'thresholds': thresholds,
            'fscore_area_trans': fscore_area_trans,
            'fscore_area_rot': fscore_area_rot
        }

    def save_results(ref_file, est_file, fscore_trans, fscore_rot, auc_result):
        est_dir = os.path.dirname(est_file)
        robustness_dir = os.path.join(est_dir, 'robustness_result')
        
        os.makedirs(robustness_dir, exist_ok=True)
        est_filename = os.path.splitext(os.path.basename(est_file))[0]
        result_file = os.path.join(robustness_dir, f'robustness_results_{est_filename}.csv')

        with open(result_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            writer.writerow(['Estimated File', est_file])
            writer.writerow([])  
            writer.writerow(['Thresholds', 'F-score (Trans)', 'F-score (Rot)'])
            
            for t, ft, fr in zip(auc_result['thresholds'], 
                                auc_result['fscore_transes'], 
                                auc_result['fscore_rots']):
                writer.writerow([f'{t:.3f}', f'{ft:.4f}', f'{fr:.4f}'])
            
            writer.writerow([])  
            writer.writerow(['AUC (Trans)', f"{auc_result['fscore_area_trans']:.4f}"])
            writer.writerow(['AUC (Rot)', f"{auc_result['fscore_area_rot']:.4f}"])

        print(f"Results saved to: {result_file}")

    def plot_robustness_metrics(auc_result, ref_file, est_file):
        
        
        thresholds = auc_result['thresholds']
        fscore_transes = auc_result['fscore_transes']
        fscore_rots = auc_result['fscore_rots']
        
        
        thresholds_array = np.array(thresholds)
        fscore_transes_array = np.array(fscore_transes)
        fscore_rots_array = np.array(fscore_rots)
        
        
        sorted_indices = np.argsort(thresholds_array)[::-1]
        thresholds_sorted = thresholds_array[sorted_indices]
        fscore_transes_sorted = fscore_transes_array[sorted_indices]
        fscore_rots_sorted = fscore_rots_array[sorted_indices]
        
        
        fine_thresholds = np.linspace(thresholds_sorted.max(), thresholds_sorted.min(), 5000)
        
        
        f_trans_interp = interp1d(thresholds_sorted, fscore_transes_sorted, kind='cubic')
        f_rot_interp = interp1d(thresholds_sorted, fscore_rots_sorted, kind='cubic')
        
        
        smooth_fscore_trans = f_trans_interp(fine_thresholds)
        smooth_fscore_rot = f_rot_interp(fine_thresholds)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(fine_thresholds, smooth_fscore_trans, 
                label=f'Translation $R_p$ [AUC: {auc_result["fscore_area_trans"]:.3f}]', 
                color='#1f77b4', linestyle='-', linewidth=2)
        plt.plot(fine_thresholds, smooth_fscore_rot, 
                label=f'Rotation $R_r$ [AUC: {auc_result["fscore_area_rot"]:.3f}]', 
                color='#ff7f0e', linestyle='-', linewidth=2)
        
        plt.xlabel('Threshold', fontsize=14)
        plt.ylabel('F1 score', fontsize=14)
        plt.title('Robustness Metric', fontweight='bold', fontsize=16)
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.xlim(min(fine_thresholds), max(fine_thresholds))
        plt.ylim(0, 1.01)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        
        output_dir = os.path.dirname(ref_file)
        output_filename = f"robustness_plot_{os.path.basename(est_file).split('.')[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.show()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        plt.close()
