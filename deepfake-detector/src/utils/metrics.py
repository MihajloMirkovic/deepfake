import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from typing import Dict, List, Tuple
from collections import defaultdict


def aggregate_predictions(
    frame_logits: List[float],
    video_ids: List[str],
    method: str = 'mean'
) -> Tuple[List[str], np.ndarray]:
    """
    Aggregate frame-level predictions to video-level predictions.

    Args:
        frame_logits: List of frame-level logits
        video_ids: List of video IDs corresponding to each frame
        method: Aggregation method ('mean' or 'median')

    Returns:
        Tuple of (unique_video_ids, video_level_scores)
    """
    # Group logits by video ID
    video_logits = defaultdict(list)
    for logit, video_id in zip(frame_logits, video_ids):
        video_logits[video_id].append(logit)

    # Aggregate
    unique_video_ids = []
    video_scores = []

    for video_id in sorted(video_logits.keys()):
        logits = np.array(video_logits[video_id])

        if method == 'mean':
            score = np.mean(logits)
        elif method == 'median':
            score = np.median(logits)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        unique_video_ids.append(video_id)
        video_scores.append(score)

    return unique_video_ids, np.array(video_scores)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_scores: Predicted probabilities/scores

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['true_negatives'] = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    metrics['false_positives'] = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    metrics['false_negatives'] = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    metrics['true_positives'] = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    # ROC-AUC and AP (if we have scores)
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    except:
        metrics['roc_auc'] = 0.0

    try:
        metrics['average_precision'] = average_precision_score(y_true, y_scores)
    except:
        metrics['average_precision'] = 0.0

    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    output_path: str,
    video_ids: List[str] = None
) -> pd.DataFrame:
    """
    Generate and save a detailed classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Predicted scores/probabilities
        output_path: Path to save the report CSV
        video_ids: Optional list of video IDs

    Returns:
        DataFrame with the report
    """
    # Calculate overall metrics
    metrics = calculate_metrics(y_true, y_pred, y_scores)

    # Create summary DataFrame
    summary_df = pd.DataFrame([metrics])

    # Create per-sample DataFrame if video_ids provided
    if video_ids is not None:
        samples_df = pd.DataFrame({
            'video_id': video_ids,
            'true_label': y_true,
            'predicted_label': y_pred,
            'score': y_scores
        })

        # Save both to CSV
        with open(output_path, 'w') as f:
            f.write("# Overall Metrics\n")
            summary_df.to_csv(f, index=False)
            f.write("\n# Per-Video Predictions\n")
            samples_df.to_csv(f, index=False)
    else:
        summary_df.to_csv(output_path, index=False)

    print(f"Classification report saved to {output_path}")
    return summary_df


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for printing (e.g., 'Train', 'Val', 'Test')
    """
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    if 'roc_auc' in metrics and metrics['roc_auc'] > 0:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    if 'average_precision' in metrics and metrics['average_precision'] > 0:
        print(f"  AP:        {metrics['average_precision']:.4f}")

    # Print confusion matrix
    if all(k in metrics for k in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']):
        print("\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
        print(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")


def get_video_labels(video_ids: List[str], frame_labels: List[int]) -> Dict[str, int]:
    """
    Extract video-level labels from frame-level labels.

    Args:
        video_ids: List of video IDs
        frame_labels: List of frame labels

    Returns:
        Dictionary mapping video_id to label
    """
    video_labels = {}
    for video_id, label in zip(video_ids, frame_labels):
        if video_id not in video_labels:
            video_labels[video_id] = label
        else:
            # Sanity check: all frames from same video should have same label
            assert video_labels[video_id] == label, \
                f"Inconsistent labels for video {video_id}"

    return video_labels
