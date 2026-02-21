#!/usr/bin/env python3
"""
Hybrid Chart Generator for Phase Evolution and Baseline Model Comparison

Generates two separate figures:
1. Phase evolution (Phase 1 → Phase 6.5) at AP@8px (strict threshold)
2. Phase 6.5 at AP@15px vs baseline models (fair external comparison)

Extracts metrics automatically from:
- Phase report.txt files (image/compare/phase*/report.txt)
- Baseline model JSON files (image/compare/*/evaluation_*.json)

Supported baseline models:
- Faster RCNN (Detectron2)
- RetinaNet (Detectron2)
- YOLOv26s
"""

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class MetricsExtractor:
    """Extracts metrics from phase reports and baseline model JSONs."""
    
    def __init__(self, base_dir: Path = Path("image/compare")):
        self.base_dir = base_dir

    @staticmethod
    def _scale_detectron2_ap(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return value
        # Detectron2 reports AP in percent (e.g., 1.55 = 1.55%)
        return value / 100.0 if value > 1.0 else value
    
    def extract_phase_metrics(self, phase_name: str) -> Optional[Dict]:
        """
        Extract metrics from phase report.txt file.
        
        Returns dict with keys: tp, fp, fn, precision, recall, f1, ap
        """
        report_path = self.base_dir / phase_name / "report.txt"
        if not report_path.exists():
            print(f"⚠️  {report_path} not found")
            return None
        
        try:
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Extract total line: "Total TP=... FP=... FN=..."
            # And metrics line: "Precision=... Recall=... F1=... AP=..."
            total_match = re.search(
                r'Total TP=(\d+) FP=(\d+) FN=(\d+)',
                content
            )
            metrics_match = re.search(
                r'Precision=([\d.]+) Recall=([\d.]+) F1=([\d.]+) AP=([\d.]+)',
                content
            )
            
            if total_match and metrics_match:
                return {
                    'tp': int(total_match.group(1)),
                    'fp': int(total_match.group(2)),
                    'fn': int(total_match.group(3)),
                    'precision': float(metrics_match.group(1)),
                    'recall': float(metrics_match.group(2)),
                    'f1': float(metrics_match.group(3)),
                    'ap': float(metrics_match.group(4)),
                }
        except Exception as e:
            print(f"❌ Error parsing {report_path}: {e}")
            return None
    
    def extract_baseline_metrics_rgb(self, model_name: str) -> Optional[Dict]:
        """
        Extract metrics from baseline model RGB JSON.
        
        Handles two JSON formats:
        1. Detectron2 format (nested with 'coco_metrics' and 'custom_metrics')
        2. YOLOv26s format (flat structure)
        
        Args:
            model_name: e.g., 'detectron2-faster-rcnn-fpn-3x' or 'yolov26s'
        
        Returns dict with keys: tp, fp, fn, precision, recall, f1_score, ap
        """
        json_path = self.base_dir / model_name / "evaluation_rgb.json"
        if not json_path.exists():
            # Try alternate naming convention
            json_path = self.base_dir / model_name / "evaluation_metrics_rgb.json"
        
        if not json_path.exists():
            print(f"⚠️  RGB JSON not found for {model_name}")
            return None
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Try Detectron2 format first
            if 'custom_metrics' in data:
                metrics = data.get('custom_metrics', {})
                ap = self._scale_detectron2_ap(data.get('coco_metrics', {}).get('AP'))
                ap50 = self._scale_detectron2_ap(data.get('coco_metrics', {}).get('AP50'))
                return {
                    'tp': metrics.get('tp'),
                    'fp': metrics.get('fp'),
                    'fn': metrics.get('fn'),
                    'precision': metrics.get('precision'),
                    'recall': metrics.get('recall'),
                    'f1': metrics.get('f1_score'),
                    'ap': ap,
                    'ap50': ap50,
                    'mean_confidence': metrics.get('mean_confidence'),
                }
            # Try YOLOv26s flat format
            elif 'tp' in data or 'TP' in data:
                return {
                    'tp': data.get('tp') or data.get('TP'),
                    'fp': data.get('fp') or data.get('FP'),
                    'fn': data.get('fn') or data.get('FN'),
                    'precision': data.get('precision') or data.get('Precision'),
                    'recall': data.get('recall') or data.get('Recall'),
                    'f1': data.get('f1_score') or data.get('F1'),
                    'ap': data.get('ap') or data.get('AP'),
                    'ap50': data.get('ap50') or data.get('AP50'),
                    'mean_confidence': data.get('mean_confidence') or data.get('mean_conf'),
                }
            else:
                print(f"⚠️  Unknown JSON format in {json_path}")
                print(f"   Available keys: {list(data.keys())[:10]}")
                return None
        except Exception as e:
            print(f"❌ Error parsing {json_path}: {e}")
            return None
    
    def extract_baseline_metrics_thermal(self, model_name: str) -> Optional[Dict]:
        """
        Extract metrics from baseline model Thermal JSON.
        
        Handles two JSON formats:
        1. Detectron2 format (nested with 'coco_metrics' and 'custom_metrics')
        2. YOLOv26s format (flat structure)
        """
        json_path = self.base_dir / model_name / "evaluation_thermal.json"
        if not json_path.exists():
            # Try alternate naming convention
            json_path = self.base_dir / model_name / "evaluation_metrics_thermal.json"
        
        if not json_path.exists():
            return None
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Try Detectron2 format first
            if 'custom_metrics' in data:
                metrics = data.get('custom_metrics', {})
                ap = self._scale_detectron2_ap(data.get('coco_metrics', {}).get('AP'))
                return {
                    'tp': metrics.get('tp'),
                    'fp': metrics.get('fp'),
                    'fn': metrics.get('fn'),
                    'precision': metrics.get('precision'),
                    'recall': metrics.get('recall'),
                    'f1': metrics.get('f1_score'),
                    'ap': ap,
                    'mean_confidence': metrics.get('mean_confidence'),
                }
            # Try YOLOv26s flat format
            elif 'tp' in data or 'TP' in data:
                return {
                    'tp': data.get('tp') or data.get('TP'),
                    'fp': data.get('fp') or data.get('FP'),
                    'fn': data.get('fn') or data.get('FN'),
                    'precision': data.get('precision') or data.get('Precision'),
                    'recall': data.get('recall') or data.get('Recall'),
                    'f1': data.get('f1_score') or data.get('F1'),
                    'ap': data.get('ap') or data.get('AP'),
                    'mean_confidence': data.get('mean_confidence') or data.get('mean_conf'),
                }
            else:
                return None
        except Exception as e:
            print(f"❌ Error parsing {json_path}: {e}")
            return None


class HybridChartGenerator:
    """Generates hybrid charts for phase evolution and baseline comparison."""
    
    def __init__(self):
        self.phases_8px = [
            'phase1', 'phase2', 'phase3',
            'phase6.1', 'phase6.2', 'phase6.3', 'phase6.5'
        ]
        self.baselines = {
            'Faster RCNN': 'detectron2-faster-rcnn-fpn-3x',
            'RetinaNet': 'detectron2-retinanet-fpn-1x',
            'YOLOv26s': 'yolov26s',
        }
        self.extractor = MetricsExtractor()
    
    def extract_all_metrics(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Extract all metrics.
        
        Returns: (phases_metrics, baselines_rgb, baselines_thermal, phase6.5_ap15)
        """
        print("📊 Extracting phase metrics (AP@8px)...")
        phases_metrics = {}
        for phase in self.phases_8px:
            metrics = self.extractor.extract_phase_metrics(phase)
            if metrics:
                phases_metrics[phase] = metrics
                print(f"  ✓ {phase}: AP={metrics['ap']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        
        print("\n📊 Extracting baseline metrics (RGB, AP@15px)...")
        baselines_rgb = {}
        for name, model_dir in self.baselines.items():
            metrics = self.extractor.extract_baseline_metrics_rgb(model_dir)
            if metrics:
                baselines_rgb[name] = metrics
                print(f"  ✓ {name} (RGB): AP={metrics['ap']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        
        print("\n📊 Extracting baseline metrics (Thermal, AP@15px)...")
        baselines_thermal = {}
        for name, model_dir in self.baselines.items():
            metrics = self.extractor.extract_baseline_metrics_thermal(model_dir)
            if metrics:
                baselines_thermal[name] = metrics
                print(f"  ✓ {name} (Thermal): AP={metrics['ap']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        
        print("\n📊 Extracting Phase 6.5 AP@15px metrics...")
        phase6_5_ap15 = self.extractor.extract_phase_metrics('phase6.5_AP15')
        if phase6_5_ap15:
            print(f"  ✓ Phase 6.5 (AP@15px): AP={phase6_5_ap15['ap']:.4f}, P={phase6_5_ap15['precision']:.4f}, R={phase6_5_ap15['recall']:.4f}")
        
        return phases_metrics, baselines_rgb, baselines_thermal, phase6_5_ap15
    
    def generate_charts(self, output_path: str = "comparison_charts.png"):
        """Generate two charts and save to separate files."""
        phases_metrics, baselines_rgb, baselines_thermal, phase6_5_ap15 = self.extract_all_metrics()
        
        if not phases_metrics:
            print("❌ No phase metrics found!")
            return
        
        output = Path(output_path)
        suffix = output.suffix if output.suffix else ".png"
        stem = output.stem if output.suffix else output.name
        phase_path = output.with_name(f"{stem}_phase{suffix}")
        baseline_path = output.with_name(f"{stem}_baseline{suffix}")

        # ===== FIGURE 1: Phase Evolution (AP@8px) =====
        fig_left, ax_left = plt.subplots(1, 1, figsize=(9, 7))
        
        # ===== LEFT PANEL: Phase Evolution (AP@8px) =====
        phase_names = list(phases_metrics.keys())
        phase_ap = [float(phases_metrics[p]['ap']) if phases_metrics[p]['ap'] else 0 for p in phase_names]
        phase_precision = [float(phases_metrics[p]['precision']) if phases_metrics[p]['precision'] else 0 for p in phase_names]
        phase_recall = [float(phases_metrics[p]['recall']) if phases_metrics[p]['recall'] else 0 for p in phase_names]
        phase_f1 = [float(phases_metrics[p]['f1']) if phases_metrics[p]['f1'] else 0 for p in phase_names]
        
        # Use full phase labels
        phase_display = phase_names
        x_pos = np.arange(len(phase_names))
        width = 0.2
        
        bars1_left = ax_left.bar(x_pos - 1.5*width, phase_ap, width, label='AP', color='#1f77b4', alpha=0.8)
        bars2_left = ax_left.bar(x_pos - 0.5*width, phase_precision, width, label='Precision', color='#ff7f0e', alpha=0.8)
        bars3_left = ax_left.bar(x_pos + 0.5*width, phase_recall, width, label='Recall', color='#2ca02c', alpha=0.8)
        bars4_left = ax_left.bar(x_pos + 1.5*width, phase_f1, width, label='F1', color='#d62728', alpha=0.8)
        
        ax_left.set_xlabel('Phase', fontsize=12, fontweight='bold')
        ax_left.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax_left.set_title('Phase Evolution (AP@8px)', fontsize=13, fontweight='bold')
        ax_left.set_xticks(x_pos)
        ax_left.set_xticklabels(phase_display, rotation=30, ha='right')
        ax_left.set_ylim(0, 1.05)
        ax_left.legend(loc='upper left', fontsize=10)
        ax_left.grid(axis='y', alpha=0.3)
        
        # Add value labels on each bar with staggered offsets
        label_offsets = [0.012, 0.024, 0.036, 0.048]
        for bars, offset in zip([bars1_left, bars2_left, bars3_left, bars4_left], label_offsets):
            for bar in bars:
                height = bar.get_height()
                ax_left.text(bar.get_x() + bar.get_width() / 2, height + offset,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=6)
        
        plt.tight_layout()
        plt.savefig(phase_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Phase evolution chart saved to {phase_path}")

        # ===== FIGURE 2: Baseline Comparison (AP@15px) =====
        if baselines_rgb and phase6_5_ap15:
            fig_right, ax_right = plt.subplots(1, 1, figsize=(12, 7))
            # Combine RGB and Thermal baselines and Phase 6.5
            model_names = []
            ap_values = []
            precision_values = []
            recall_values = []
            f1_values = []
            
            # Add RGB baselines
            for model_name, metrics in baselines_rgb.items():
                model_names.append(f'{model_name} (RGB)')
                ap_values.append(float(metrics['ap']) if metrics['ap'] else 0)
                precision_values.append(float(metrics['precision']) if metrics['precision'] else 0)
                recall_values.append(float(metrics['recall']) if metrics['recall'] else 0)
                f1_values.append(float(metrics['f1']) if metrics['f1'] else 0)
            
            # Add Thermal baselines if available
            for model_name, metrics in baselines_thermal.items():
                model_names.append(f'{model_name} (Thermal)')
                ap_values.append(float(metrics['ap']) if metrics['ap'] else 0)
                precision_values.append(float(metrics['precision']) if metrics['precision'] else 0)
                recall_values.append(float(metrics['recall']) if metrics['recall'] else 0)
                f1_values.append(float(metrics['f1']) if metrics['f1'] else 0)
            
            # Add Phase 6.5 (RGBT)
            phase6_base_idx = len(model_names)
            model_names.append('Phase 6.5 (RGBT)')
            ap_values.append(float(phase6_5_ap15['ap']) if phase6_5_ap15['ap'] else 0)
            precision_values.append(float(phase6_5_ap15['precision']) if phase6_5_ap15['precision'] else 0)
            recall_values.append(float(phase6_5_ap15['recall']) if phase6_5_ap15['recall'] else 0)
            f1_values.append(float(phase6_5_ap15['f1']) if phase6_5_ap15['f1'] else 0)
            
            x_pos_right = np.arange(len(model_names))
            
            bars1 = ax_right.bar(x_pos_right - 1.5*width, ap_values, width, label='AP', 
                                color='#1f77b4', alpha=0.8)
            bars2 = ax_right.bar(x_pos_right - 0.5*width, precision_values, width, label='Precision', 
                                color='#ff7f0e', alpha=0.8)
            bars3 = ax_right.bar(x_pos_right + 0.5*width, recall_values, width, label='Recall', 
                                color='#2ca02c', alpha=0.8)
            bars4 = ax_right.bar(x_pos_right + 1.5*width, f1_values, width, label='F1', 
                                color='#d62728', alpha=0.8)
            
            # Highlight Phase 6.5 and RGB baselines differently
            for idx in range(len(model_names)):
                is_phase6 = idx == phase6_base_idx
                is_thermal = idx >= len(baselines_rgb) and idx < phase6_base_idx
                
                if is_phase6:
                    # Bold border for Phase 6.5
                    for bars in [bars1, bars2, bars3, bars4]:
                        bars[idx].set_edgecolor('black')
                        bars[idx].set_linewidth(2.5)
                elif is_thermal:
                    # Dashed border for Thermal
                    for bars in [bars1, bars2, bars3, bars4]:
                        bars[idx].set_edgecolor('gray')
                        bars[idx].set_linewidth(1.5)
                        bars[idx].set_linestyle('--')
            
            ax_right.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax_right.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax_right.set_title('Baseline Comparison (AP@15px)', fontsize=13, fontweight='bold')
            ax_right.set_xticks(x_pos_right)
            ax_right.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax_right.set_ylim(0, 1.05)
            ax_right.legend(loc='upper left', fontsize=10)
            ax_right.grid(axis='y', alpha=0.3)
            
            # Add value labels on each bar with staggered offsets
            for bars, offset in zip([bars1, bars2, bars3, bars4], label_offsets):
                for bar in bars:
                    height = bar.get_height()
                    ax_right.text(bar.get_x() + bar.get_width() / 2, height + offset,
                                  f'{height:.3f}', ha='center', va='bottom', fontsize=6)
            
            # Add custom legend for highlights
            black_patch = mpatches.Patch(facecolor='none', edgecolor='black', linewidth=2.5, label='Phase 6.5 (RGBT)')
            if baselines_thermal:
                gray_patch = mpatches.Patch(facecolor='none', edgecolor='gray', linewidth=1.5, label='Thermal baseline')
                handles, labels = ax_right.get_legend_handles_labels()
                ax_right.legend(handles + [black_patch, gray_patch], labels + ['Phase 6.5 (RGBT)', 'Thermal baseline'], 
                               loc='upper left', fontsize=10)
            else:
                handles, labels = ax_right.get_legend_handles_labels()
                ax_right.legend(handles + [black_patch], labels + ['Phase 6.5 (RGBT)'], 
                               loc='upper left', fontsize=10)
            plt.tight_layout()
            plt.savefig(baseline_path, dpi=300, bbox_inches='tight')
            print(f"✅ Baseline comparison chart saved to {baseline_path}")
            return fig_left, fig_right

        print("⚠️  Baseline comparison chart not generated (missing baselines or phase6.5_AP15 metrics).")
        return fig_left, None
    
    def print_comparison_table(self):
        """Print detailed comparison table to console."""
        phases_metrics, baselines_rgb, baselines_thermal, phase6_5_ap15 = self.extract_all_metrics()
        
        print("\n" + "="*100)
        print("TABLE 1: PHASE EVOLUTION (AP@8px - Internal Rigorous Evaluation)")
        print("="*100)
        print(f"{'Phase':<15} {'TP':>8} {'FP':>8} {'FN':>8} {'Precision':>12} {'Recall':>12} {'F1':>10} {'AP':>10}")
        print("-"*100)
        
        for phase in self.phases_8px:
            if phase in phases_metrics:
                m = phases_metrics[phase]
                print(f"{phase:<15} {m['tp']:>8} {m['fp']:>8} {m['fn']:>8} "
                      f"{m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1']:>10.4f} {m['ap']:>10.4f}")
        
        print("\n" + "="*100)
        print("TABLE 2: BASELINE COMPARISON (AP@15px - Fair External Evaluation)")
        print("="*100)
        print(f"{'Model':<25} {'Dataset':<15} {'AP':>10} {'Precision':>12} {'Recall':>12} {'F1':>10}")
        print("-"*100)
        
        if baselines_rgb:
            for model_name, metrics in baselines_rgb.items():
                if metrics:
                    ap = metrics.get('ap') or 0
                    prec = metrics.get('precision') or 0
                    rec = metrics.get('recall') or 0
                    f1 = metrics.get('f1') or 0
                    if ap != 0 and prec != 0 and rec != 0:
                        print(f"{model_name:<25} {'RGB':<15} {float(ap):>10.4f} "
                              f"{float(prec):>12.4f} {float(rec):>12.4f} {float(f1):>10.4f}")
        
        if baselines_thermal:
            for model_name, metrics in baselines_thermal.items():
                if metrics:
                    ap = metrics.get('ap') or 0
                    prec = metrics.get('precision') or 0
                    rec = metrics.get('recall') or 0
                    f1 = metrics.get('f1') or 0
                    if ap != 0 and prec != 0 and rec != 0:
                        print(f"{model_name:<25} {'Thermal':<15} {float(ap):>10.4f} "
                              f"{float(prec):>12.4f} {float(rec):>12.4f} {float(f1):>10.4f}")
        
        if phase6_5_ap15:
            print(f"{'Phase 6.5 (RGBT)':<25} {'RGBT':<15} {phase6_5_ap15['ap']:>10.4f} "
                  f"{phase6_5_ap15['precision']:>12.4f} {phase6_5_ap15['recall']:>12.4f} {phase6_5_ap15['f1']:>10.4f}")
        
        print("\n" + "="*100)
        print("TABLE 3: PHASE 6.5 THRESHOLD IMPACT")
        print("="*100)
        print(f"{'Threshold':<15} {'TP':>8} {'FP':>8} {'FN':>8} {'Precision':>12} {'Recall':>12} {'AP':>10}")
        print("-"*100)
        
        if 'phase6.5' in phases_metrics and phase6_5_ap15:
            m_8px = phases_metrics['phase6.5']
            m_15px = phase6_5_ap15
            print(f"{'8px (Strict)':<15} {m_8px['tp']:>8} {m_8px['fp']:>8} {m_8px['fn']:>8} "
                  f"{m_8px['precision']:>12.4f} {m_8px['recall']:>12.4f} {m_8px['ap']:>10.4f}")
            print(f"{'15px (Lenient)':<15} {m_15px['tp']:>8} {m_15px['fp']:>8} {m_15px['fn']:>8} "
                  f"{m_15px['precision']:>12.4f} {m_15px['recall']:>12.4f} {m_15px['ap']:>10.4f}")
            
            delta_tp = m_15px['tp'] - m_8px['tp']
            delta_fp = m_15px['fp'] - m_8px['fp']
            delta_fn = m_15px['fn'] - m_8px['fn']
            delta_prec = m_15px['precision'] - m_8px['precision']
            delta_rec = m_15px['recall'] - m_8px['recall']
            delta_ap = m_15px['ap'] - m_8px['ap']
            
            print(f"{'Delta':<15} {delta_tp:>+8} {delta_fp:>+8} {delta_fn:>+8} "
                  f"{delta_prec:>+12.4f} {delta_rec:>+12.4f} {delta_ap:>+10.4f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate charts for phase evolution and baseline comparison'
    )
    parser.add_argument(
        '--output', '-o',
        default='comparison_charts.png',
        help='Base output path for charts (default: comparison_charts.png)'
    )
    parser.add_argument(
        '--table', '-t',
        action='store_true',
        help='Print detailed comparison tables to console'
    )
    
    args = parser.parse_args()
    
    generator = HybridChartGenerator()
    
    # Always generate charts with the specified output path
    generator.generate_charts(args.output)
    
    # Additionally print tables if requested
    if args.table:
        print()  # Add spacing before tables
        generator.print_comparison_table()


if __name__ == '__main__':
    main()
