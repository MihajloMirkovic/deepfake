"""
Split FaceForensics++ dataset (custom flat structure) by source ID to prevent data leakage.
"""

import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import pandas as pd
from tqdm import tqdm


def get_source_ids_from_original(original_dir: Path) -> list:
    """Get all source IDs from original videos."""
    source_ids = []
    for video_path in original_dir.glob('*.mp4'):
        # Extract ID from filename (e.g., '003.mp4' -> '003')
        source_id = video_path.stem
        source_ids.append(source_id)
    return sorted(source_ids)


def split_source_ids(source_ids: list, train_n: int, val_n: int, test_n: int, seed: int = 42):
    """Split source IDs into train/val/test sets."""
    random.seed(seed)
    source_ids_copy = list(source_ids)
    random.shuffle(source_ids_copy)

    total_needed = train_n + val_n + test_n
    if len(source_ids_copy) < total_needed:
        print(f"Warning: Only {len(source_ids_copy)} source IDs available, but {total_needed} requested")
        print(f"Adjusting split proportionally...")

        total = len(source_ids_copy)
        ratio = total / total_needed
        train_n = int(train_n * ratio)
        val_n = int(val_n * ratio)
        test_n = total - train_n - val_n

    splits = {
        'train': set(source_ids_copy[:train_n]),
        'val': set(source_ids_copy[train_n:train_n + val_n]),
        'test': set(source_ids_copy[train_n + val_n:train_n + val_n + test_n])
    }

    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])} source IDs")
    print(f"  Val:   {len(splits['val'])} source IDs")
    print(f"  Test:  {len(splits['test'])} source IDs")

    return splits


def copy_videos(dataset_root: Path, splits: dict, output_dir: Path):
    """
    Copy videos to output directory based on splits.
    """
    manifest_data = []

    # Process each split
    for split_name, source_id_set in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")

        # Create output directories
        real_output = output_dir / split_name / 'REAL'
        fake_output = output_dir / split_name / 'FAKE'
        real_output.mkdir(parents=True, exist_ok=True)
        fake_output.mkdir(parents=True, exist_ok=True)

        # Copy original (REAL) videos
        original_dir = dataset_root / 'original'
        if original_dir.exists():
            print(f"\nCopying REAL videos...")
            real_count = 0
            for video_path in tqdm(list(original_dir.glob('*.mp4'))):
                source_id = video_path.stem
                if source_id in source_id_set:
                    dest = real_output / video_path.name
                    shutil.copy2(video_path, dest)
                    real_count += 1

                    manifest_data.append({
                        'video_path': str(dest.relative_to(output_dir)),
                        'split': split_name,
                        'label': 'REAL',
                        'source_id': source_id,
                        'manipulation_type': 'original'
                    })
            print(f"  Copied {real_count} REAL videos")

        # Copy manipulated (FAKE) videos
        manipulation_methods = ['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures', 'FaceShifter']

        for method in manipulation_methods:
            method_dir = dataset_root / method
            if not method_dir.exists():
                print(f"  Skipping {method} (not found)")
                continue

            print(f"\nCopying FAKE videos from {method}...")
            fake_count = 0

            for video_path in tqdm(list(method_dir.glob('*.mp4'))):
                # Extract both source IDs from filename (e.g., '003_012.mp4')
                stem = video_path.stem
                parts = stem.split('_')

                if len(parts) >= 2:
                    source_id_1 = parts[0]
                    source_id_2 = parts[1]

                    # Include video if EITHER source ID is in this split
                    # This ensures we test generalization to new combinations
                    if source_id_1 in source_id_set or source_id_2 in source_id_set:
                        dest = fake_output / video_path.name
                        shutil.copy2(video_path, dest)
                        fake_count += 1

                        # Use the primary source ID (first one)
                        manifest_data.append({
                            'video_path': str(dest.relative_to(output_dir)),
                            'split': split_name,
                            'label': 'FAKE',
                            'source_id': f"{source_id_1}_{source_id_2}",
                            'manipulation_type': method
                        })

            print(f"  Copied {fake_count} FAKE videos from {method}")

    # Save manifest
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = output_dir / 'split_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved manifest to {manifest_path}")
    print(f"{'='*60}")

    # Print statistics
    print("\nDataset statistics:")
    stats = manifest_df.groupby(['split', 'label']).size().reset_index(name='count')
    print(stats.to_string(index=False))

    print("\nManipulation methods per split:")
    method_stats = manifest_df[manifest_df['label'] == 'FAKE'].groupby(['split', 'manipulation_type']).size().reset_index(name='count')
    print(method_stats.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Split FaceForensics++ dataset by source ID (custom structure)"
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Root directory containing original/, Deepfakes/, etc.'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/raw_videos',
        help='Output directory for split videos'
    )
    parser.add_argument(
        '--train-ids',
        type=int,
        default=600,
        help='Number of source IDs for training (default: 600)'
    )
    parser.add_argument(
        '--val-ids',
        type=int,
        default=200,
        help='Number of source IDs for validation (default: 200)'
    )
    parser.add_argument(
        '--test-ids',
        type=int,
        default=200,
        help='Number of source IDs for testing (default: 200)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.out)

    if not dataset_root.exists():
        print(f"Error: Dataset root directory not found: {dataset_root}")
        return

    # Check for original directory
    original_dir = dataset_root / 'original'
    if not original_dir.exists():
        print(f"Error: Original videos directory not found: {original_dir}")
        return

    # Get all source IDs from original videos
    print("Collecting source IDs from original videos...")
    source_ids = get_source_ids_from_original(original_dir)
    print(f"Found {len(source_ids)} source IDs (from {source_ids[0]} to {source_ids[-1]})")

    # Split source IDs
    splits = split_source_ids(
        source_ids,
        args.train_ids,
        args.val_ids,
        args.test_ids,
        args.seed
    )

    # Copy videos to output directory
    copy_videos(dataset_root, splits, output_dir)

    print("\n" + "="*60)
    print("Dataset split completed successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
