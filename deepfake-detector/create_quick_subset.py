"""
Create a small subset of videos for quick testing.
"""

import shutil
from pathlib import Path
import random

random.seed(42)

# Source and destination
source_dir = Path('data/raw_videos')
dest_dir = Path('data/raw_videos_subset')

# Number of videos to copy per split
subset_config = {
    'train': {'REAL': 50, 'FAKE': 100},  # 20 from each of 5 manipulation methods
    'val': {'REAL': 20, 'FAKE': 40},
    'test': {'REAL': 30, 'FAKE': 60}
}

print("Creating quick test subset...")
print("=" * 60)

total_copied = 0

for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()} split:")

    for label in ['REAL', 'FAKE']:
        source_path = source_dir / split / label
        dest_path = dest_dir / split / label
        dest_path.mkdir(parents=True, exist_ok=True)

        # Get all videos
        all_videos = list(source_path.glob('*.mp4'))

        # Select random subset
        n_videos = subset_config[split][label]
        selected_videos = random.sample(all_videos, min(n_videos, len(all_videos)))

        # Copy videos
        for video in selected_videos:
            dest_file = dest_path / video.name
            if not dest_file.exists():
                shutil.copy2(video, dest_file)

        print(f"  {label}: {len(selected_videos)} videos")
        total_copied += len(selected_videos)

print("\n" + "=" * 60)
print(f"Total videos in subset: {total_copied}")
print(f"Subset created in: {dest_dir}")
print("=" * 60)
