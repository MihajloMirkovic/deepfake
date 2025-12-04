"""
Crop and align faces from extracted frames using MTCNN.
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import cv2
import numpy as np


def get_face_detector(device: str = 'cpu'):
    """
    Initialize MTCNN face detector.

    Args:
        device: Device to run detector on ('cpu', 'cuda', or 'mps')

    Returns:
        MTCNN detector
    """
    try:
        from facenet_pytorch import MTCNN

        # Convert device string to torch.device
        if device == 'cuda' and torch.cuda.is_available():
            torch_device = torch.device('cuda')
        elif device == 'mps' and torch.backends.mps.is_available():
            torch_device = torch.device('mps')
        else:
            torch_device = torch.device('cpu')

        print(f"Using device: {torch_device}")

        # Initialize MTCNN
        mtcnn = MTCNN(
            image_size=224,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=torch_device,
            keep_all=False  # Only keep the largest face
        )

        return mtcnn

    except ImportError:
        print("Error: facenet-pytorch not installed.")
        print("Install with: pip install facenet-pytorch")
        sys.exit(1)


def crop_face_from_image(
    image_path: Path,
    output_path: Path,
    mtcnn,
    target_size: int = 224
) -> bool:
    """
    Detect and crop face from an image.

    Args:
        image_path: Path to input image
        output_path: Path to save cropped face
        mtcnn: MTCNN detector
        target_size: Target size for cropped face

    Returns:
        True if face was detected and cropped, False otherwise
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Detect face
        img_cropped = mtcnn(img)

        if img_cropped is not None:
            # Convert tensor to PIL Image
            # MTCNN returns tensor in range [-1, 1], convert to [0, 255]
            img_cropped = (img_cropped.permute(1, 2, 0).cpu().numpy() * 128 + 127.5).astype(np.uint8)
            img_cropped = Image.fromarray(img_cropped)

            # Resize to target size (if needed)
            if img_cropped.size != (target_size, target_size):
                img_cropped = img_cropped.resize((target_size, target_size), Image.BILINEAR)

            # Save
            img_cropped.save(output_path, quality=95)
            return True
        else:
            return False

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def process_directory(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 224,
    device: str = 'cpu',
    overwrite: bool = False
):
    """
    Process all frames in a directory structure.

    Args:
        input_dir: Input directory containing frames
        output_dir: Output directory for cropped faces
        target_size: Target size for cropped faces
        device: Device to use for face detection
        overwrite: Whether to overwrite input files or create new output
    """
    # Initialize face detector
    print("Initializing face detector...")
    mtcnn = get_face_detector(device)

    # If not overwriting, create output directory structure
    if not overwrite and output_dir != input_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
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

            # Get all image files
            image_files = list(label_dir.glob('*.jpg')) + list(label_dir.glob('*.png'))

            if not image_files:
                print(f"  No images found in {label_dir}")
                continue

            print(f"\n  Processing {label_name} images ({len(image_files)} files)...")

            # Output directory
            if overwrite:
                output_label_dir = label_dir
            else:
                output_label_dir = output_dir / split_name / label_name
                output_label_dir.mkdir(parents=True, exist_ok=True)

            # Process each image
            success_count = 0
            failed_images = []

            for image_path in tqdm(image_files, desc=f"  {split_name}/{label_name}"):
                output_path = output_label_dir / image_path.name

                # Skip if output already exists (unless overwriting)
                if output_path.exists() and not overwrite:
                    success_count += 1
                    continue

                # Crop face
                success = crop_face_from_image(
                    image_path,
                    output_path,
                    mtcnn,
                    target_size
                )

                if success:
                    success_count += 1
                else:
                    failed_images.append(image_path.name)
                    # Remove the input frame if no face was detected
                    if overwrite and output_path.exists():
                        output_path.unlink()

            print(f"    Successfully cropped {success_count}/{len(image_files)} images")

            if failed_images:
                print(f"    Failed images (no face detected): {len(failed_images)}")
                # Show first few failures
                for img in failed_images[:5]:
                    print(f"      - {img}")
                if len(failed_images) > 5:
                    print(f"      ... and {len(failed_images) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Crop faces from frames using MTCNN"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/frames',
        help='Input directory containing frames'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for cropped faces (default: overwrite input)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='Target size for cropped faces (default: 224)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (default: cpu)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)

    # If output not specified, overwrite input
    if args.output is None:
        output_dir = input_dir
        overwrite = True
        print("Note: Will overwrite input frames with cropped faces")
    else:
        output_dir = Path(args.output)
        overwrite = False

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Auto-detect best device if not specified
    if args.device == 'cpu':
        if torch.cuda.is_available():
            device = 'cuda'
            print("CUDA available, using GPU")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("MPS available, using Apple Silicon GPU")
        else:
            device = 'cpu'
            print("Using CPU (this will be slow)")
    else:
        device = args.device

    print(f"\nCropping faces from frames...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Size:   {args.size}x{args.size}")

    process_directory(
        input_dir,
        output_dir,
        target_size=args.size,
        device=device,
        overwrite=overwrite
    )

    print("\nDone! Face cropping completed.")


if __name__ == '__main__':
    main()
