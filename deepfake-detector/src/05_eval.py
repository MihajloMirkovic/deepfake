"""
Evaluation script for deepfake detection model.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.dataset import get_dataloaders
from utils.model import get_model, load_checkpoint
from utils.metrics import (
    aggregate_predictions,
    calculate_metrics,
    print_metrics,
    generate_classification_report,
    get_video_labels
)


def evaluate_model(
    model: nn.Module,
    dataloader,
    device,
    aggregation: str = 'mean',
    threshold: float = 0.5
):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        aggregation: Method to aggregate frame predictions ('mean' or 'median')
        threshold: Classification threshold

    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()

    all_frame_logits = []
    all_frame_labels = []
    all_video_ids = []

    print("Running inference...")
    with torch.no_grad():
        for images, labels, video_ids in tqdm(dataloader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Store frame-level predictions
            logits = outputs.cpu().numpy().flatten()
            all_frame_logits.extend(logits.tolist())
            all_frame_labels.extend(labels.numpy().flatten().tolist())
            all_video_ids.extend(video_ids)

    # Aggregate to video-level
    print(f"Aggregating frame predictions to video-level (method: {aggregation})...")
    video_labels_dict = get_video_labels(all_video_ids, all_frame_labels)
    unique_video_ids, video_scores = aggregate_predictions(
        all_frame_logits,
        all_video_ids,
        method=aggregation
    )

    # Get video-level labels
    video_labels = np.array([video_labels_dict[vid] for vid in unique_video_ids])

    # Convert logits to probabilities
    video_probs = 1 / (1 + np.exp(-video_scores))

    # Predict labels
    video_preds = (video_probs > threshold).astype(int)

    # Calculate metrics
    metrics = calculate_metrics(video_labels, video_preds, video_probs)

    return {
        'metrics': metrics,
        'video_ids': unique_video_ids,
        'true_labels': video_labels,
        'pred_labels': video_preds,
        'scores': video_probs
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/frames',
        help='Path to data directory'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/metrics',
        help='Output directory for metrics'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu/cuda/mps)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for evaluation'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override batch size if specified
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']

    # Get device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Create dataloaders
    print("\nLoading datasets...")
    dataloaders = get_dataloaders(
        data_dir=args.data,
        batch_size=batch_size,
        num_workers=4 if device.type == 'cuda' else 0,
        image_size=config['preprocessing']['face_size']
    )

    # Check if data is available
    if len(dataloaders[args.split].dataset) == 0:
        print(f"Error: No {args.split} data found!")
        sys.exit(1)

    # Create model
    print("\nCreating model...")
    model = get_model(
        name=config['model']['name'],
        pretrained=False,  # We'll load trained weights
        num_classes=config['model']['num_classes']
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    results = evaluate_model(
        model,
        dataloaders[args.split],
        device,
        aggregation=config['evaluation']['aggregation'],
        threshold=config['evaluation']['threshold']
    )

    # Print metrics
    print_metrics(results['metrics'], prefix=args.split.capitalize())

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f'{args.split}_report.csv'
    generate_classification_report(
        results['true_labels'],
        results['pred_labels'],
        results['scores'],
        str(report_path),
        video_ids=results['video_ids']
    )

    print(f"\nResults saved to {report_path}")
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
