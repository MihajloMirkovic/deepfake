"""
Test model robustness to video compression.
Re-encode test videos with different CRF values and evaluate performance.
"""

import argparse
import sys
import subprocess
from pathlib import Path
import yaml
import shutil
import json
import torch
from tqdm import tqdm
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.dataset import DeepfakeDataset, get_transforms
from utils.model import get_model, load_checkpoint
from utils.metrics import (
    aggregate_predictions,
    calculate_metrics,
    print_metrics,
    get_video_labels
)


def compress_video(input_path: Path, output_path: Path, crf: int) -> bool:
    """
    Compress video using ffmpeg with specified CRF value.

    Args:
        input_path: Path to input video
        output_path: Path to save compressed video
        crf: CRF value (0-51, lower = better quality)

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', 'medium',
            '-y',  # Overwrite output file
            str(output_path)
        ]

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error compressing {input_path}: {e}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False


def extract_frames_from_video(video_path: Path, output_dir: Path, fps: int = 3, max_frames: int = 16):
    """Extract frames from a compressed video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30

    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1

    frame_count = 0
    extracted_count = 0
    video_name = video_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0 and extracted_count < max_frames:
            frame_filename = f"{video_name}_frame_{extracted_count:04d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_count += 1

        frame_count += 1
        if extracted_count >= max_frames:
            break

    cap.release()
    return extracted_count


def crop_face(image_path: Path, output_path: Path, mtcnn, target_size: int = 224):
    """Crop face from image using MTCNN."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert('RGB')
        img_cropped = mtcnn(img)

        if img_cropped is not None:
            img_cropped = (img_cropped.permute(1, 2, 0).cpu().numpy() * 128 + 127.5).astype(np.uint8)
            img_cropped = Image.fromarray(img_cropped)

            if img_cropped.size != (target_size, target_size):
                img_cropped = img_cropped.resize((target_size, target_size), Image.BILINEAR)

            img_cropped.save(output_path, quality=95)
            return True
        return False
    except Exception as e:
        return False


def evaluate_compressed_videos(
    model,
    frames_dir: Path,
    device,
    image_size: int = 224,
    aggregation: str = 'mean'
):
    """Evaluate model on compressed video frames."""
    model.eval()

    # Create dataset from frames
    transform = get_transforms(split='test', image_size=image_size)
    dataset = DeepfakeDataset(str(frames_dir.parent), split=frames_dir.name, transform=transform)

    if len(dataset) == 0:
        print(f"Warning: No frames found in {frames_dir}")
        return None

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    all_frame_logits = []
    all_frame_labels = []
    all_video_ids = []

    with torch.no_grad():
        for images, labels, video_ids in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            logits = outputs.cpu().numpy().flatten()
            all_frame_logits.extend(logits.tolist())
            all_frame_labels.extend(labels.numpy().flatten().tolist())
            all_video_ids.extend(video_ids)

    # Aggregate to video-level
    video_labels_dict = get_video_labels(all_video_ids, all_frame_labels)
    unique_video_ids, video_scores = aggregate_predictions(
        all_frame_logits,
        all_video_ids,
        method=aggregation
    )

    video_labels = [video_labels_dict[vid] for vid in unique_video_ids]
    video_probs = 1 / (1 + torch.tensor(video_scores).exp().numpy())
    video_preds = (video_probs > 0.5).astype(int)

    metrics = calculate_metrics(video_labels, video_preds, video_probs)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test model on compressed videos")
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw_videos/test',
        help='Input directory with test videos'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/compression_test',
        help='Output directory for compressed videos and results'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--crf',
        type=int,
        nargs='+',
        default=[18, 28, 35],
        help='CRF values to test'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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

    # Create model
    print("\nLoading model...")
    model = get_model(
        name=config['model']['name'],
        pretrained=False,
        num_classes=config['model']['num_classes']
    )
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # Initialize face detector
    print("Initializing face detector...")
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(
            image_size=224,
            margin=0,
            device=device,
            keep_all=False
        )
    except ImportError:
        print("Error: facenet-pytorch not installed")
        sys.exit(1)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Test each CRF value
    for crf in args.crf:
        print(f"\n{'='*60}")
        print(f"Testing CRF = {crf}")
        print(f"{'='*60}")

        crf_dir = output_dir / f"crf_{crf}"
        crf_videos_dir = crf_dir / 'videos'
        crf_frames_dir = crf_dir / 'frames' / 'test'

        # Compress videos
        print("\nCompressing videos...")
        for label in ['REAL', 'FAKE']:
            label_input_dir = input_dir / label
            if not label_input_dir.exists():
                continue

            label_output_dir = crf_videos_dir / label
            label_output_dir.mkdir(parents=True, exist_ok=True)

            videos = list(label_input_dir.glob('*.mp4'))
            for video in tqdm(videos, desc=f"Compressing {label}"):
                output_video = label_output_dir / video.name
                compress_video(video, output_video, crf)

        # Extract frames
        print("\nExtracting frames...")
        for label in ['REAL', 'FAKE']:
            label_videos_dir = crf_videos_dir / label
            if not label_videos_dir.exists():
                continue

            label_frames_dir = crf_frames_dir / label
            label_frames_dir.mkdir(parents=True, exist_ok=True)

            videos = list(label_videos_dir.glob('*.mp4'))
            for video in tqdm(videos, desc=f"Extracting {label}"):
                extract_frames_from_video(
                    video,
                    label_frames_dir,
                    fps=config['preprocessing']['fps'],
                    max_frames=config['preprocessing']['max_frames_per_video']
                )

        # Crop faces
        print("\nCropping faces...")
        for label in ['REAL', 'FAKE']:
            label_frames_dir = crf_frames_dir / label
            if not label_frames_dir.exists():
                continue

            frames = list(label_frames_dir.glob('*.jpg'))
            for frame in tqdm(frames, desc=f"Cropping {label}"):
                temp_output = frame.parent / f"temp_{frame.name}"
                if crop_face(frame, temp_output, mtcnn, 224):
                    shutil.move(str(temp_output), str(frame))
                else:
                    # Remove frame if no face detected
                    frame.unlink()

        # Evaluate
        print("\nEvaluating...")
        metrics = evaluate_compressed_videos(
            model,
            crf_frames_dir,
            device,
            image_size=config['preprocessing']['face_size'],
            aggregation=config['evaluation']['aggregation']
        )

        if metrics:
            print_metrics(metrics, prefix=f"CRF {crf}")
            results[f"crf_{crf}"] = metrics

    # Save results
    results_path = output_dir / 'compression_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("\nCompression test completed!")


if __name__ == '__main__':
    main()
