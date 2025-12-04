"""
Script to download FaceForensics++ dataset.
This is an optional helper script - you can also manually download the dataset.
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm


def download_with_kagglehub(output_dir: str):
    """
    Download FaceForensics++ dataset using kagglehub.

    Args:
        output_dir: Directory to save the dataset
    """
    try:
        import kagglehub
        print("Downloading FaceForensics++ from Kaggle...")
        print("Note: This requires Kaggle API credentials.")
        print("Please configure: ~/.kaggle/kaggle.json")

        # Download the dataset
        path = kagglehub.dataset_download("sorokin/faceforensics")
        print(f"Dataset downloaded to: {path}")

        # Optionally, you can move files to output_dir
        print(f"You may want to move files from {path} to {output_dir}")

    except ImportError:
        print("Error: kagglehub not installed. Install with: pip install kagglehub")
    except Exception as e:
        print(f"Error downloading dataset: {e}")


def download_with_gdown(file_id: str, output_path: str):
    """
    Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID
        output_path: Path to save the downloaded file
    """
    try:
        import gdown
        print(f"Downloading file from Google Drive...")

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

        print(f"Downloaded to: {output_path}")

    except ImportError:
        print("Error: gdown not installed. Install with: pip install gdown")
    except Exception as e:
        print(f"Error downloading file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download FaceForensics++ dataset"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw_videos',
        help='Output directory for downloaded data'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['kaggle', 'gdrive', 'manual'],
        default='manual',
        help='Download method to use'
    )
    parser.add_argument(
        '--gdrive-id',
        type=str,
        help='Google Drive file ID (if using gdrive method)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    if args.method == 'kaggle':
        download_with_kagglehub(args.output)

    elif args.method == 'gdrive':
        if not args.gdrive_id:
            print("Error: --gdrive-id required when using gdrive method")
            return

        output_file = os.path.join(args.output, 'faceforensics.zip')
        download_with_gdown(args.gdrive_id, output_file)

        print("\nTo extract the downloaded file:")
        print(f"  unzip {output_file} -d {args.output}")

    else:  # manual
        print("Manual download instructions:")
        print("\n1. Official FaceForensics++ repository:")
        print("   https://github.com/ondyari/FaceForensics")
        print("\n2. Request access and download the dataset")
        print("\n3. Alternative datasets:")
        print("   - Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics")
        print("   - DFDC: https://www.kaggle.com/c/deepfake-detection-challenge")
        print(f"\n4. Place downloaded videos in: {args.output}")
        print("\nExpected structure:")
        print(f"  {args.output}/")
        print("    original_sequences/")
        print("      youtube/")
        print("        c23/videos/  (or c40)")
        print("    manipulated_sequences/")
        print("      Deepfakes/")
        print("        c23/videos/")
        print("      FaceSwap/")
        print("        c23/videos/")
        print("      Face2Face/")
        print("        c23/videos/")
        print("      NeuralTextures/")
        print("        c23/videos/")


if __name__ == '__main__':
    main()
