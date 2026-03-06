#!/usr/bin/env python3
"""
Calculate AP and generate PR-curves from evaluation data.

This script supports two input formats:
1. JSON files with custom_metrics/pr_curve data (Detectron2, YOLO baselines)
2. CSV files with score labels (Phase 6.x models)

Usage:
    # For JSON evaluation files
    python calculate_ap_pr_curve.py --json evaluation_metrics.json
    
    # For CSV score files (Phase models)
    python calculate_ap_pr_curve.py --csv scores.csv --gt-count 54391
    
    # Generate PR curve plot
    python calculate_ap_pr_curve.py --json evaluation_metrics.json --plot output.png
    
    # Compare multiple models
    python calculate_ap_pr_curve.py --compare model1.json model2.json --plot comparison.png
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def calculate_ap_from_arrays(precision, recall):
    """
    Calculate Average Precision using all-point interpolation.
    
    Args:
        precision: Array of precision values
        recall: Array of recall values
        
    Returns:
        float: Average Precision (area under PR curve)
    """
    precision = np.array(precision)
    recall = np.array(recall)
    
    # Sort by recall
    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]
    
    # Add sentinel values at boundaries
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute precision envelope (maximum precision at each recall level)
    # This ensures the curve is monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Calculate area under curve using trapezoidal integration
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    
    return ap


def load_json_evaluation(json_path):
    """
    Load evaluation data from JSON file (Detectron2 or YOLO format).
    
    Args:
        json_path: Path to JSON evaluation file
        
    Returns:
        dict: Contains 'precision', 'recall', 'ap', 'tp', 'fp', 'fn'
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if this is YOLO format (has pr_curve with recalls/precisions)
    if 'pr_curve' in data and 'recalls' in data['pr_curve']:
        pr_curve = data['pr_curve']
        precision = np.array(pr_curve['precisions'])
        recall = np.array(pr_curve['recalls'])
        
        # Calculate AP from PR curve
        ap = calculate_ap_from_arrays(precision, recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'tp': data.get('tp'),
            'fp': data.get('fp'),
            'fn': data.get('gt_boxes', 0) - data.get('tp', 0),
            'final_precision': data.get('precision'),
            'final_recall': data.get('recall')
        }
    
    # Check if this is Detectron2 format (has custom_metrics.pr_curve)
    elif 'custom_metrics' in data and 'pr_curve' in data['custom_metrics']:
        pr_curve = data['custom_metrics']['pr_curve']
        precision = np.array(pr_curve['precision'])
        recall = np.array(pr_curve['recall'])
        
        # Calculate AP from PR curve
        ap = calculate_ap_from_arrays(precision, recall)
        
        custom = data['custom_metrics']
        return {
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'tp': custom.get('tp'),
            'fp': custom.get('fp'),
            'fn': custom.get('fn'),
            'final_precision': custom.get('precision'),
            'final_recall': custom.get('recall')
        }
    
    else:
        raise ValueError(f"Unknown JSON format in {json_path}")


def load_csv_evaluation(csv_path, gt_count):
    """
    Load evaluation data from CSV scores file (Phase model format).
    
    Args:
        csv_path: Path to scores.csv file
        gt_count: Total number of ground truth objects
        
    Returns:
        dict: Contains 'precision', 'recall', 'ap', 'tp', 'fp', 'fn'
    """
    # Load scores CSV
    scores = pd.read_csv(csv_path)
    
    # Sort by score descending
    scores = scores.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Calculate cumulative TP and FP
    tp_cumsum = (scores['label'] == 'TP').astype(int).cumsum()
    fp_cumsum = (scores['label'] == 'FP').astype(int).cumsum()
    
    total_tp = (scores['label'] == 'TP').sum()
    total_fp = (scores['label'] == 'FP').sum()
    
    # Calculate precision and recall at each threshold
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / gt_count
    
    # Calculate AP
    ap = calculate_ap_from_arrays(precision.values, recall.values)
    
    # Calculate final metrics at optimal threshold (max F1)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_idx = f1_scores.idxmax()
    
    return {
        'precision': precision.values,
        'recall': recall.values,
        'ap': ap,
        'tp': total_tp,
        'fp': total_fp,
        'fn': gt_count - total_tp,
        'final_precision': precision.iloc[best_idx],
        'final_recall': recall.iloc[best_idx],
        'scores': scores['score'].values
    }


def plot_pr_curve(results_dict, output_path, title="Precision-Recall Curve"):
    """
    Plot PR curves for one or multiple models.
    
    Args:
        results_dict: Dictionary mapping model names to result dictionaries
        output_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#2E86AB', '#F18F01', '#A23B72', '#6A994E', '#BC4749', 
              '#386641', '#8B2635', '#577590', '#90A955']
    
    for idx, (name, results) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        auc_value = results['ap']  # AUC = Area Under PR Curve
        label = f"{name} (AUC={auc_value:.4f})"
        plt.plot(results['recall'], results['precision'], 
                linewidth=2, label=label, color=color)
    
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ PR curve plot saved to: {output_path}")


def print_results(name, results):
    """Print evaluation results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Results for: {name}")
    print(f"{'='*60}")
    print(f"Average Precision (AP):  {results['ap']:.4f}")
    
    if results.get('final_precision') is not None:
        print(f"\nOptimal Operating Point:")
        print(f"  Precision:             {results['final_precision']:.4f}")
        print(f"  Recall:                {results['final_recall']:.4f}")
        f1 = 2 * results['final_precision'] * results['final_recall'] / \
             (results['final_precision'] + results['final_recall'])
        print(f"  F1-Score:              {f1:.4f}")
    
    if results.get('tp') is not None:
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (TP):   {results['tp']:,}")
        print(f"  False Positives (FP):  {results['fp']:,}")
        print(f"  False Negatives (FN):  {results['fn']:,}")
        
        total_gt = results['tp'] + results['fn']
        max_recall = results['tp'] / total_gt if total_gt > 0 else 0
        print(f"\nMaximum Achievable Recall: {max_recall:.4f} ({max_recall*100:.2f}%)")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate AP and generate PR-curves from evaluation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    parser.add_argument('--json', type=str, help='Path to JSON evaluation file')
    parser.add_argument('--csv', type=str, help='Path to CSV scores file')
    parser.add_argument('--gt-count', type=int, default=54391,
                       help='Ground truth object count (for CSV mode, default: 54391)')
    
    # Comparison mode
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Compare multiple models (provide paths to JSON/CSV files)')
    parser.add_argument('--names', nargs='+', type=str,
                       help='Custom names for compared models')
    
    # Output options
    parser.add_argument('--plot', type=str, help='Save PR curve plot to file')
    parser.add_argument('--title', type=str, default='Precision-Recall Curve',
                       help='Plot title')
    parser.add_argument('--output-json', type=str, 
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    results_dict = {}
    
    # Single model evaluation
    if args.json:
        results = load_json_evaluation(args.json)
        name = Path(args.json).stem
        results_dict[name] = results
        print_results(name, results)
    
    elif args.csv:
        results = load_csv_evaluation(args.csv, args.gt_count)
        name = Path(args.csv).stem
        results_dict[name] = results
        print_results(name, results)
    
    # Comparison mode
    elif args.compare:
        for idx, filepath in enumerate(args.compare):
            filepath = Path(filepath)
            
            # Determine name
            if args.names and idx < len(args.names):
                name = args.names[idx]
            else:
                name = filepath.stem
            
            # Load data based on file extension
            if filepath.suffix == '.json':
                results = load_json_evaluation(filepath)
            elif filepath.suffix == '.csv':
                results = load_csv_evaluation(filepath, args.gt_count)
            else:
                print(f"Warning: Unknown file type {filepath}, skipping")
                continue
            
            results_dict[name] = results
            print_results(name, results)
    
    else:
        parser.print_help()
        return
    
    # Generate plot if requested
    if args.plot and results_dict:
        plot_pr_curve(results_dict, args.plot, args.title)
    
    # Save results to JSON if requested
    if args.output_json and results_dict:
        output_data = {}
        for name, results in results_dict.items():
            output_data[name] = {
                'ap': float(results['ap']),
                'final_precision': float(results.get('final_precision', 0)),
                'final_recall': float(results.get('final_recall', 0)),
                'tp': int(results.get('tp', 0)),
                'fp': int(results.get('fp', 0)),
                'fn': int(results.get('fn', 0))
            }
        
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to: {args.output_json}")


if __name__ == '__main__':
    main()
