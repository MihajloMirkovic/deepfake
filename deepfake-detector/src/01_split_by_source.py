"""
Split FaceForensics++ dataset by source ID to prevent data leakage.
This ensures that the same source video doesn't appear in both train and test sets.
"""

import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import pandas as pd
from tqdm import tqdm


def extract_source_id(video_name: str) -> str:
    """
    Extract source ID from video filename.
    E.g., '003_012.mp4' -> '003' (target video ID)
         '012_003.mp4' -> '012' (source video ID for same pair)

    For FaceForensics++, we consider both IDs to ensure no leakage.
    Return a tuple or combined identifier.
    """
    stem = Path(video_name).stem
    parts = stem.split('_')

    if len(parts) >= 2:
        # Both IDs should stay together (they're a pair)
        return f"{parts[0]}_{parts[1]}"
    else:
        return parts[0]


def collect_videos(ffpp_root: Path) -> dict:
    """
    Collect all videos from FaceForensics++ directory structure.

    Returns:
        Dictionary with structure:
        {
            'original': {source_id: [video_paths]},
            'Deepfakes': {source_id: [video_paths]},
            'FaceSwap': {source_id: [video_paths]},
            ...
        }
    """
    videos_by_method = {}

    # Collect original videos
    original_dir = ffpp_root / 'original_sequences' / 'youtube' / 'c23' / 'videos'
    if not original_dir.exists():
        # Try c40 compression
        original_dir = ffpp_root / 'original_sequences' / 'youtube' / 'c40' / 'videos'

    if original_dir.exists():
        original_videos = defaultdict(list)
        for video_path in original_dir.glob('*.mp4'):
            source_id = extract_source_id(video_path.name)
            original_videos[source_id].append(video_path)
        videos_by_method['original'] = original_videos
        print(f"Found {len(original_videos)} unique source IDs in original videos")

    # Collect manipulated videos
    manipulated_dir = ffpp_root / 'manipulated_sequences'
    if manipulated_dir.exists():
        for method_dir in manipulated_dir.iterdir():
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name
            videos_dir = method_dir / 'c23' / 'videos'
            if not videos_dir.exists():
                videos_dir = method_dir / 'c40' / 'videos'

            if videos_dir.exists():
                method_videos = defaultdict(list)
                for video_path in videos_dir.glob('*.mp4'):
                    source_id = extract_source_id(video_path.name)
                    method_videos[source_id].append(video_path)
                videos_by_method[method_name] = method_videos
                print(f"Found {len(method_videos)} unique source IDs in {method_name}")

    return videos_by_method


def split_source_ids(source_ids: list, train_n: int, val_n: int, test_n: int, seed: int = 42):
    """
    Split source IDs into train/val/test sets.

    Args:
        source_ids: List of unique source IDs
        train_n: Number of source IDs for training
        val_n: Number of source IDs for validation
        test_n: Number of source IDs for testing
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' lists of source IDs
    """
    random.seed(seed)
    source_ids = list(source_ids)
    random.shuffle(source_ids)

    total_needed = train_n + val_n + test_n
    if len(source_ids) < total_needed:
        print(f"Warning: Only {len(source_ids)} source IDs available, but {total_needed} requested")
        print(f"Adjusting split proportionally...")

        # Adjust proportionally
        total = len(source_ids)
        ratio = total / total_needed
        train_n = int(train_n * ratio)
        val_n = int(val_n * ratio)
        test_n = total - train_n - val_n

    splits = {
        'train': source_ids[:train_n],
        'val': source_ids[train_n:train_n + val_n],
        'test': source_ids[train_n + val_n:train_n + val_n + test_n]
    }

    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])} source IDs")
    print(f"  Val:   {len(splits['val'])} source IDs")
    print(f"  Test:  {len(splits['test'])} source IDs")

    return splits


def copy_videos(videos_by_method: dict, splits: dict, output_dir: Path):
    """
    Copy videos to output directory based on splits.

    Args:
        videos_by_method: Videos organized by manipulation method
        splits: Dictionary of source ID splits
        output_dir: Output directory
    """
    manifest_data = []

    for split_name, source_ids in splits.items():
        print(f"\nProcessing {split_name} split...")
        source_id_set = set(source_ids)

        # Copy original videos (REAL)
        if 'original' in videos_by_method:
            real_output = output_dir / split_name / 'REAL'
            real_output.mkdir(parents=True, exist_ok=True)

            for source_id, video_paths in tqdm(videos_by_method['original'].items()):
                if source_id in source_id_set:
                    for video_path in video_paths:
                        dest = real_output / video_path.name
                        shutil.copy2(video_path, dest)

                        manifest_data.append({
                            'video_path': str(dest.relative_to(output_dir)),
                            'split': split_name,
                            'label': 'REAL',
                            'source_id': source_id,
                            'manipulation_type': 'original'
                        })

        # Copy manipulated videos (FAKE)
        fake_output = output_dir / split_name / 'FAKE'
        fake_output.mkdir(parents=True, exist_ok=True)

        for method_name, method_videos in videos_by_method.items():
            if method_name == 'original':
                continue

            # For test set, include additional methods for generalization testing
            if split_name == 'test' or method_name in ['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures']:
                for source_id, video_paths in tqdm(method_videos.items(), desc=method_name):
                    if source_id in source_id_set:
                        for video_path in video_paths:
                            dest = fake_output / video_path.name
                            shutil.copy2(video_path, dest)

                            manifest_data.append({
                                'video_path': str(dest.relative_to(output_dir)),
                                'split': split_name,
                                'label': 'FAKE',
                                'source_id': source_id,
                                'manipulation_type': method_name
                            })

    # Save manifest
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = output_dir / 'split_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nSaved manifest to {manifest_path}")

    # Print statistics
    print("\nDataset statistics:")
    print(manifest_df.groupby(['split', 'label']).size())


def main():
    parser = argparse.ArgumentParser(
        description="Split FaceForensics++ dataset by source ID to prevent data leakage"
    )
    parser.add_argument(
        '--ffpp-root',
        type=str,
        required=True,
        help='Root directory of FaceForensics++ dataset'
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
        default=100,
        help='Number of source IDs for training'
    )
    parser.add_argument(
        '--val-ids',
        type=int,
        default=50,
        help='Number of source IDs for validation'
    )
    parser.add_argument(
        '--test-ids',
        type=int,
        default=60,
        help='Number of source IDs for testing'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    ffpp_root = Path(args.ffpp_root)
    output_dir = Path(args.out)

    if not ffpp_root.exists():
        print(f"Error: FaceForensics++ root directory not found: {ffpp_root}")
        return

    # Collect all videos
    print("Collecting videos from FaceForensics++ directory...")
    videos_by_method = collect_videos(ffpp_root)

    if not videos_by_method:
        print("Error: No videos found. Check the directory structure.")
        return

    # Get all unique source IDs from original videos
    if 'original' not in videos_by_method:
        print("Error: No original videos found")
        return

    all_source_ids = list(videos_by_method['original'].keys())
    print(f"\nTotal unique source IDs: {len(all_source_ids)}")

    # Split source IDs
    splits = split_source_ids(
        all_source_ids,
        args.train_ids,
        args.val_ids,
        args.test_ids,
        args.seed
    )

    # Copy videos to output directory
    copy_videos(videos_by_method, splits, output_dir)

    print("\nDone! Dataset split completed successfully.")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
