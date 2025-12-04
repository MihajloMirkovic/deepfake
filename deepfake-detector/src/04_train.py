"""
Training script for deepfake detection model.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.dataset import get_dataloaders, collate_fn
from utils.model import get_model, save_checkpoint, count_parameters
from utils.metrics import (
    aggregate_predictions,
    calculate_metrics,
    print_metrics,
    get_video_labels
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch: int
):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for images, labels, video_ids in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape: (batch_size, 1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        # Store predictions
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs.flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_labels = (all_preds > 0.5).astype(int)

    metrics = calculate_metrics(all_labels, pred_labels, all_preds)
    metrics['loss'] = epoch_loss

    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device,
    epoch: int,
    aggregation: str = 'mean'
):
    """Validate for one epoch with video-level aggregation."""
    model.eval()
    running_loss = 0.0

    all_frame_logits = []
    all_frame_labels = []
    all_video_ids = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for images, labels, video_ids in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # Store frame-level predictions
            logits = outputs.cpu().numpy().flatten()
            all_frame_logits.extend(logits.tolist())
            all_frame_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_video_ids.extend(video_ids)

            pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)

    # Aggregate to video-level
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
    video_preds = (video_probs > 0.5).astype(int)

    # Calculate metrics
    metrics = calculate_metrics(video_labels, video_preds, video_probs)
    metrics['loss'] = epoch_loss

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
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
        '--output',
        type=str,
        default='outputs',
        help='Output directory for models and logs'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu/cuda/mps)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    # Create output directories
    output_dir = Path(args.output)
    models_dir = output_dir / 'models'
    metrics_dir = output_dir / 'metrics'
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nLoading datasets...")
    dataloaders = get_dataloaders(
        data_dir=args.data,
        batch_size=config['training']['batch_size'],
        num_workers=4 if device.type == 'cuda' else 0,
        image_size=config['preprocessing']['face_size']
    )

    # Check if data is available
    if len(dataloaders['train'].dataset) == 0:
        print("Error: No training data found!")
        sys.exit(1)

    # Create model
    print("\nCreating model...")
    model = get_model(
        name=config['model']['name'],
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes']
    )
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Calculate class weights for imbalanced data
    train_dataset = dataloaders['train'].dataset
    num_real = sum(1 for sample in train_dataset.samples if sample['label'] == 0)
    num_fake = sum(1 for sample in train_dataset.samples if sample['label'] == 1)
    pos_weight = num_real / num_fake if num_fake > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)

    print(f"\nClass distribution:")
    print(f"  REAL: {num_real}")
    print(f"  FAKE: {num_fake}")
    print(f"  Positive weight: {pos_weight:.2f}")

    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )

    # Training loop
    print("\nStarting training...")
    best_f1 = 0.0
    epochs_without_improvement = 0
    training_history = []

    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch
        )
        print_metrics(train_metrics, prefix="Train")

        # Validate
        val_metrics = validate_epoch(
            model, dataloaders['val'], criterion, device, epoch,
            aggregation=config['evaluation']['aggregation']
        )
        print_metrics(val_metrics, prefix="Validation")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nLearning rate: {current_lr:.6f}")

        # Save training history
        training_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr
        })

        # Save best model based on validation F1
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                str(models_dir / 'best.pt')
            )
            print(f"Saved new best model (F1: {best_f1:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping after {epoch} epochs (no improvement for {epochs_without_improvement} epochs)")
            break

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        epoch,
        val_metrics,
        str(models_dir / 'final.pt')
    )

    # Save training history
    history_path = metrics_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print("\nTraining completed!")
    print(f"Best validation F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
