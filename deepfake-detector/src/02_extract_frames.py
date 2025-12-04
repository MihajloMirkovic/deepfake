"""
Extract frames from videos at specified FPS.
"""

import argparse
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import sys


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps: int = 3,
    max_frames: int = 16
) -> int:
    """
    Extract frames from a single video.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract
        max_frames: Maximum number of frames to extract

    Returns:
        Number of frames extracted
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps == 0:
        print(f"Warning: Could not get FPS for {video_path}, using default 30")
        video_fps = 30

    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1

    # Extract frames
    frame_count = 0
    extracted_count = 0
    video_name = video_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Extract frame at specified interval
            if frame_count % frame_interval == 0 and extracted_count < max_frames:
                # Save frame
                frame_filename = f"{video_name}_frame_{extracted_count:04d}.jpg"
                frame_path = output_dir / frame_filename

                # Save with high quality
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_count += 1

            frame_count += 1

            # Stop if we've extracted enough frames
            if extracted_count >= max_frames:
                break

    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")

    finally:
        cap.release()

    return extracted_count


def process_directory(
    input_dir: Path,
    output_dir: Path,
    fps: int = 3,
    max_frames: int = 16
):
    """
    Process all videos in a directory (preserving REAL/FAKE structure).

    Args:
        input_dir: Input directory containing videos
        output_dir: Output directory for frames
        fps: Frames per second to extract
        max_frames: Maximum frames per video
    """
    # Process each split (train/val/test)
    for split_dir in input_dir.iterdir():
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        print(f"\nProcessing {split_name} split...")

        # Process REAL and FAKE subdirectories
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label_name = label_dir.name
            if label_name not in ['REAL', 'FAKE']:
                continue

            # Get all video files
            video_files = list(label_dir.glob('*.mp4')) + list(label_dir.glob('*.avi'))

            if not video_files:
                print(f"  No videos found in {label_dir}")
                continue

            print(f"\n  Processing {label_name} videos ({len(video_files)} files)...")

            # Output directory for this split/label
            frames_output_dir = output_dir / split_name / label_name

            # Process each video
            total_extracted = 0
            failed_videos = []

            for video_path in tqdm(video_files, desc=f"  {split_name}/{label_name}"):
                try:
                    num_frames = extract_frames_from_video(
                        video_path,
                        frames_output_dir,
                        fps=fps,
                        max_frames=max_frames
                    )
                    total_extracted += num_frames

                    if num_frames == 0:
                        failed_videos.append(video_path.name)

                except Exception as e:
                    print(f"    Error processing {video_path.name}: {e}")
                    failed_videos.append(video_path.name)

            print(f"    Extracted {total_extracted} frames from {len(video_files)} videos")

            if failed_videos:
                print(f"    Failed videos ({len(failed_videos)}):")
                for vid in failed_videos[:10]:  # Show first 10
                    print(f"      - {vid}")
                if len(failed_videos) > 10:
                    print(f"      ... and {len(failed_videos) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw_videos',
        help='Input directory containing videos'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/frames',
        help='Output directory for extracted frames'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=3,
        help='Frames per second to extract (default: 3)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=16,
        help='Maximum frames per video (default: 16)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"Extracting frames from videos...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  FPS:    {args.fps}")
    print(f"  Max frames per video: {args.max_frames}")

    process_directory(input_dir, output_dir, fps=args.fps, max_frames=args.max_frames)

    print("\nDone! Frames extracted successfully.")


if __name__ == '__main__':
    main()
