"""
Generate visualization plots for model evaluation.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def parse_report_csv(csv_path: Path) -> dict:
    """
    Parse a classification report CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        Dictionary with metrics
    """
    if not csv_path.exists():
        return None

    # Read the CSV - metrics are in the first section
    df = pd.read_csv(csv_path, comment='#', nrows=1)
    return df.iloc[0].to_dict()


def plot_prf_bar(metrics_dir: Path, output_dir: Path):
    """
    Plot Precision, Recall, F1 bar chart comparing train/val/test.

    Args:
        metrics_dir: Directory containing metric CSV files
        output_dir: Directory to save plot
    """
    splits = ['train', 'val', 'test']
    metrics_names = ['precision', 'recall', 'f1', 'accuracy']

    data = {metric: [] for metric in metrics_names}
    available_splits = []

    # Load metrics for each split
    for split in splits:
        report_path = metrics_dir / f'{split}_report.csv'
        metrics = parse_report_csv(report_path)

        if metrics:
            available_splits.append(split.capitalize())
            for metric in metrics_names:
                data[metric].append(metrics.get(metric, 0.0))

    if not available_splits:
        print("No metric reports found for plotting")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(available_splits))
    width = 0.2

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        offset = width * (i - 1.5)
        ax.bar(x + offset, data[metric], width, label=metric.capitalize(), color=color, alpha=0.8)

    ax.set_xlabel('Dataset Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Across Splits', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_splits)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'prf_bar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved PRF bar chart to {output_path}")


def plot_confusion_matrix(metrics_dir: Path, output_dir: Path, split: str = 'test'):
    """
    Plot confusion matrix heatmap.

    Args:
        metrics_dir: Directory containing metric CSV files
        output_dir: Directory to save plot
        split: Which split to plot ('train', 'val', or 'test')
    """
    report_path = metrics_dir / f'{split}_report.csv'
    metrics = parse_report_csv(report_path)

    if not metrics:
        print(f"No {split} report found for confusion matrix")
        return

    # Extract confusion matrix values
    tn = int(metrics.get('true_negatives', 0))
    fp = int(metrics.get('false_positives', 0))
    fn = int(metrics.get('false_negatives', 0))
    tp = int(metrics.get('true_positives', 0))

    cm = np.array([[tn, fp], [fn, tp]])

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['REAL', 'FAKE'],
        yticklabels=['REAL', 'FAKE'],
        cbar_kws={'label': 'Count'},
        ax=ax,
        square=True
    )

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {split.capitalize()} Set', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved confusion matrix to {output_path}")


def plot_f1_vs_crf(compression_results_path: Path, output_dir: Path):
    """
    Plot F1 score vs CRF value.

    Args:
        compression_results_path: Path to compression_results.json
        output_dir: Directory to save plot
    """
    if not compression_results_path.exists():
        print("No compression test results found")
        return

    with open(compression_results_path, 'r') as f:
        results = json.load(f)

    crf_values = []
    f1_scores = []
    accuracies = []

    for key, metrics in sorted(results.items()):
        # Extract CRF value from key (e.g., 'crf_18' -> 18)
        crf = int(key.split('_')[1])
        crf_values.append(crf)
        f1_scores.append(metrics.get('f1', 0.0))
        accuracies.append(metrics.get('accuracy', 0.0))

    if not crf_values:
        print("No CRF data found")
        return

    # Create plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = '#2ecc71'
    ax1.set_xlabel('CRF Value (lower = better quality)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(crf_values, f1_scores, marker='o', linewidth=2, markersize=8,
             color=color1, label='F1 Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3)

    # Add accuracy on secondary y-axis
    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.set_ylabel('Accuracy', color=color2, fontsize=12, fontweight='bold')
    ax2.plot(crf_values, accuracies, marker='s', linewidth=2, markersize=8,
             color=color2, linestyle='--', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title
    ax1.set_title('Model Performance vs Video Compression', fontsize=14, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    output_path = output_dir / 'f1_vs_crf.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved F1 vs CRF plot to {output_path}")


def plot_training_curves(training_history_path: Path, output_dir: Path):
    """
    Plot training and validation loss/F1 curves.

    Args:
        training_history_path: Path to training_history.json
        output_dir: Directory to save plot
    """
    if not training_history_path.exists():
        print("No training history found")
        return

    with open(training_history_path, 'r') as f:
        history = json.load(f)

    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry['train']['loss'] for entry in history]
    val_loss = [entry['val']['loss'] for entry in history]
    train_f1 = [entry['train']['f1'] for entry in history]
    val_f1 = [entry['val']['f1'] for entry in history]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, train_loss, marker='o', label='Train Loss', linewidth=2, color='#3498db')
    ax1.plot(epochs, val_loss, marker='s', label='Val Loss', linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # F1 plot
    ax2.plot(epochs, train_f1, marker='o', label='Train F1', linewidth=2, color='#2ecc71')
    ax2.plot(epochs, val_f1, marker='s', label='Val F1', linewidth=2, color='#f39c12')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation F1 Score', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved training curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument(
        '--metrics-dir',
        type=str,
        default='outputs/metrics',
        help='Directory containing metric CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/plots',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--compression-results',
        type=str,
        default='outputs/compression_test/compression_results.json',
        help='Path to compression test results'
    )
    parser.add_argument(
        '--training-history',
        type=str,
        default='outputs/metrics/training_history.json',
        help='Path to training history JSON'
    )

    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'

    print("Generating plots...")

    # 1. PRF bar chart
    print("\n1. Creating PRF bar chart...")
    plot_prf_bar(metrics_dir, output_dir)

    # 2. Confusion matrix
    print("\n2. Creating confusion matrix...")
    plot_confusion_matrix(metrics_dir, output_dir, split='test')

    # 3. F1 vs CRF (if compression test was run)
    print("\n3. Creating F1 vs CRF plot...")
    compression_results_path = Path(args.compression_results)
    plot_f1_vs_crf(compression_results_path, output_dir)

    # 4. Training curves (if training history available)
    print("\n4. Creating training curves...")
    training_history_path = Path(args.training_history)
    plot_training_curves(training_history_path, output_dir)

    print("\n" + "="*60)
    print("Plot generation completed!")
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
