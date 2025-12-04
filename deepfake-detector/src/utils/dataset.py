import os
from pathlib import Path
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class DeepfakeDataset(Dataset):
    """Dataset for loading cropped face images for deepfake detection."""

    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir: Root directory containing train/val/test folders
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Collect all image paths and labels
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """Load all image paths with their labels and video IDs."""
        split_dir = self.root_dir / self.split

        # Load REAL images
        real_dir = split_dir / 'REAL'
        if real_dir.exists():
            for img_path in sorted(real_dir.glob('*.jpg')):
                # Extract video_id from filename (e.g., '003_012_frame_0001.jpg' -> '003_012')
                video_id = '_'.join(img_path.stem.split('_')[:2])
                self.samples.append({
                    'path': str(img_path),
                    'label': 0,  # REAL = 0
                    'video_id': video_id
                })

        # Load FAKE images
        fake_dir = split_dir / 'FAKE'
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob('*.jpg')):
                video_id = '_'.join(img_path.stem.split('_')[:2])
                self.samples.append({
                    'path': str(img_path),
                    'label': 1,  # FAKE = 1
                    'video_id': video_id
                })

        if len(self.samples) == 0:
            print(f"Warning: No samples found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, str]:
        """
        Returns:
            image: Transformed image tensor
            label: Binary label (0=REAL, 1=FAKE) as float
            video_id: Video identifier for aggregation
        """
        sample = self.samples[idx]

        # Load image
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, float(sample['label']), sample['video_id']


def get_transforms(split: str = 'train', image_size: int = 224) -> transforms.Compose:
    """
    Get appropriate transforms for the given split.

    Args:
        split: One of 'train', 'val', or 'test'
        image_size: Target image size

    Returns:
        Composed transforms
    """
    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        # Training augmentations
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for train, validation, and test splits.

    Args:
        data_dir: Root directory containing the data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        image_size: Image size for transforms

    Returns:
        Dictionary with 'train', 'val', and 'test' dataloaders
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        transform = get_transforms(split=split, image_size=image_size)
        dataset = DeepfakeDataset(
            root_dir=data_dir,
            split=split,
            transform=transform
        )

        # Shuffle only training data
        shuffle = (split == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')  # Drop last incomplete batch for training
        )

        dataloaders[split] = dataloader
        print(f"{split.capitalize()} dataset: {len(dataset)} images")

    return dataloaders


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Custom collate function to handle video IDs.

    Args:
        batch: List of (image, label, video_id) tuples

    Returns:
        Batched images, labels, and list of video IDs
    """
    images, labels, video_ids = zip(*batch)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.float32)

    return images, labels, list(video_ids)
