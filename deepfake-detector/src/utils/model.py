import torch
import torch.nn as nn
import timm
from typing import Optional


def get_model(
    name: str = 'xception',
    pretrained: bool = True,
    num_classes: int = 1
) -> nn.Module:
    """
    Create a model for deepfake detection.

    Args:
        name: Model architecture name (e.g., 'xception', 'tf_efficientnet_b0_ns')
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (1 for binary classification with BCEWithLogitsLoss)

    Returns:
        PyTorch model ready for training/inference
    """
    try:
        # Try to create the specified model
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        print(f"Successfully created {name} model (pretrained={pretrained})")

    except Exception as e:
        print(f"Failed to create {name}: {e}")
        print("Falling back to tf_efficientnet_b0_ns...")

        try:
            # Fallback to EfficientNet-B0
            model = timm.create_model(
                'tf_efficientnet_b0_ns',
                pretrained=pretrained,
                num_classes=num_classes
            )
            print(f"Successfully created tf_efficientnet_b0_ns model (pretrained={pretrained})")

        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("Using basic ResNet50 as final fallback...")

            # Final fallback to ResNet50
            model = timm.create_model(
                'resnet50',
                pretrained=pretrained,
                num_classes=num_classes
            )
            print(f"Successfully created resnet50 model (pretrained={pretrained})")

    return model


class DeepfakeDetector(nn.Module):
    """
    Wrapper model for deepfake detection with configurable backbone.
    """

    def __init__(
        self,
        backbone: str = 'xception',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            backbone: Name of the backbone architecture
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for the classifier head
        """
        super(DeepfakeDetector, self).__init__()

        # Get base model
        self.backbone = get_model(
            name=backbone,
            pretrained=pretrained,
            num_classes=1  # Binary classification
        )

        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Logits of shape (batch_size, 1)
        """
        return self.backbone(x)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """
    Load model weights from a checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load the model on

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: dict,
    checkpoint_path: str
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save (optional)
        epoch: Current epoch
        metrics: Dictionary of metrics to save
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
