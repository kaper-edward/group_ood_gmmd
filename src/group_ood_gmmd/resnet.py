"""
ResNet v1 for CIFAR/MNIST/STL-10 datasets.

Matches Keras implementation from CNC paper:
- ResNet-20: depth=20, n=3, used for MNIST and CIFAR-10
- ResNet-32: depth=32, n=5, used for STL-10

Architecture (ResNet v1):
- Conv-BN-ReLU order (pre-activation in v2)
- Stage 0: 32x32, 16 filters
- Stage 1: 16x16, 32 filters
- Stage 2: 8x8, 64 filters (for CIFAR/MNIST)
           24x24, 64 filters (for STL-10 with 96x96 input)

Layer Selection for CNC:
- Last 31% of activation (ReLU) layers
- ResNet-20: 6 layers (8x8x64 each) = 24,576 neurons
- ResNet-32 (STL-10): 10 layers (24x24x64 each) = 368,640 neurons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict


class ScaledClippedReLU(nn.Module):
    """ReLU with optional cap and output scale for CNC activation-scale probes."""

    def __init__(self, scale: float = 1.0, clip: Optional[float] = None):
        super().__init__()
        self.scale = float(scale)
        self.clip = None if clip is None else float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x, inplace=False)
        if self.clip is not None:
            x = torch.clamp(x, max=self.clip)
        if self.scale != 1.0:
            x = x * self.scale
        return x


class OptionAShortcut(nn.Module):
    """Original CIFAR ResNet option-A shortcut: stride slice plus zero-pad."""

    def __init__(self, in_planes: int, planes: int, stride: int):
        super().__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x[:, :, ::self.stride, ::self.stride] if self.stride != 1 else x
        if self.planes > self.in_planes:
            pad = self.planes - self.in_planes
            front = pad // 2
            back = pad - front
            out = F.pad(out, (0, 0, 0, 0, front, back), "constant", 0.0)
        return out


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet v1.

    Structure:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU

    Note: The activation after addition is important for CNC analysis.
    """

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
        use_bn: bool = True,
        relu_scale: float = 1.0,
        relu_clip: Optional[float] = None,
        residual_scale: float = 1.0,
        shortcut_mode: str = "projection",
    ):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.residual_scale = float(residual_scale)

        # First conv (may downsample)
        # When no BN, use bias in conv
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=not use_bn
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum, eps=bn_eps) if use_bn else nn.Identity()
        self.relu1 = ScaledClippedReLU(scale=relu_scale, clip=relu_clip)

        # Second conv (no downsample)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=not use_bn
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum, eps=bn_eps) if use_bn else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if shortcut_mode == "option_a":
                self.shortcut = OptionAShortcut(in_planes, planes, stride)
            elif shortcut_mode == "projection":
                # Linear projection shortcut (1x1 conv, no BN in Keras original)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=not use_bn),
                )
            else:
                raise ValueError(f"Unknown shortcut_mode: {shortcut_mode}")

        # Final ReLU after addition
        self.relu2 = ScaledClippedReLU(scale=relu_scale, clip=relu_clip)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning both output and intermediate activations.

        Returns:
            out: Output tensor
            activations: List of activation tensors [relu1_out, relu2_out]
        """
        activations = []

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        activations.append(out.clone())

        # Second conv block (no activation)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_scale != 1.0:
            out = out * self.residual_scale

        # Add shortcut
        out = out + self.shortcut(x)

        # Final activation
        out = self.relu2(out)
        activations.append(out.clone())

        return out, activations


class ResNetV1(nn.Module):
    """
    ResNet v1 for CIFAR/MNIST/STL-10.

    Args:
        num_blocks: List of number of blocks per stage [n, n, n]
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for MNIST, 3 for CIFAR/STL)
        input_size: Input image size (32 for CIFAR/MNIST, 96 for STL-10)
        bn_momentum: BatchNorm momentum (PyTorch default 0.1, Keras-style 0.01)
        bn_eps: BatchNorm epsilon (PyTorch default 1e-5, Keras-style 1e-3)
        use_bn: Whether to use BatchNorm layers (default True)
    """

    def __init__(
        self,
        num_blocks: List[int],
        num_classes: int = 10,
        in_channels: int = 3,
        input_size: int = 32,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
        use_bn: bool = True,
        relu_scale: float = 1.0,
        relu_clip: Optional[float] = None,
        residual_scale: float = 1.0,
        shortcut_mode: str = "projection",
    ):
        super(ResNetV1, self).__init__()

        self.in_planes = 16
        self.input_size = input_size
        self.num_blocks = num_blocks
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.use_bn = use_bn
        self.relu_scale = float(relu_scale)
        self.relu_clip = relu_clip
        self.residual_scale = float(residual_scale)
        self.shortcut_mode = shortcut_mode

        # Initial conv layer
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1,
            padding=1, bias=not use_bn
        )
        self.bn1 = nn.BatchNorm2d(16, momentum=bn_momentum, eps=bn_eps) if use_bn else nn.Identity()
        self.relu1 = ScaledClippedReLU(scale=relu_scale, clip=relu_clip)

        # Three stages with increasing filters: 16 -> 32 -> 64
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)

        # Calculate average pool size based on input
        if input_size == 32:  # CIFAR/MNIST
            pool_size = 8
        elif input_size == 96:  # STL-10
            pool_size = 24
        else:
            pool_size = input_size // 4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        # Weight initialization (matching Keras he_normal)
        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.ModuleList:
        """Create a stage with multiple residual blocks."""
        layers = nn.ModuleList()

        # First block may downsample
        layers.append(BasicBlock(
            self.in_planes, planes, stride,
            bn_momentum=self.bn_momentum, bn_eps=self.bn_eps,
            use_bn=self.use_bn,
            relu_scale=self.relu_scale,
            relu_clip=self.relu_clip,
            residual_scale=self.residual_scale,
            shortcut_mode=self.shortcut_mode,
        ))
        self.in_planes = planes

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(
                planes, planes, stride=1,
                bn_momentum=self.bn_momentum, bn_eps=self.bn_eps,
                use_bn=self.use_bn,
                relu_scale=self.relu_scale,
                relu_clip=self.relu_clip,
                residual_scale=self.residual_scale,
                shortcut_mode=self.shortcut_mode,
            ))

        return layers

    def _initialize_weights(self):
        """Initialize weights using he_normal (Keras default)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        last_stage_only: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input tensor (N, C, H, W)
            return_activations: If True, return activations from last stage
            last_stage_only: If True, return only last stage activations (default)
                            This matches paper's "last 31%" = last stage

        Returns:
            out: Output logits
            activations: List of activation tensors (only if return_activations=True)
                        For ResNet-20: 6 activations (3 blocks × 2 ReLUs) at 8×8×64
                        For ResNet-32: 10 activations (5 blocks × 2 ReLUs) at 24×24×64
        """
        stage3_activations = []

        # Initial conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Stage 1 (16 filters)
        for block in self.layer1:
            out, _ = block(out)

        # Stage 2 (32 filters)
        for block in self.layer2:
            out, _ = block(out)

        # Stage 3 (64 filters) - collect activations here (last 31%)
        for block in self.layer3:
            out, block_acts = block(out)
            if return_activations:
                stage3_activations.extend(block_acts)

        # Classifier
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        if return_activations:
            return out, stage3_activations

        return out, None

    def get_num_activation_layers(self) -> int:
        """Get total number of activation (ReLU) layers."""
        # 1 (initial) + 2 * num_blocks per stage
        total = 1
        for n in self.num_blocks:
            total += 2 * n
        return total

    def get_last_stage_layers(self) -> int:
        """Get number of activation layers in last stage (used for CNC)."""
        # Last stage has 2 activations per block
        return 2 * self.num_blocks[2]

    def get_total_neurons(self) -> int:
        """Get total neurons in last stage activations."""
        h, w, c = self.get_activation_shape()
        num_layers = self.get_last_stage_layers()
        return num_layers * h * w * c

    def get_activation_shape(self) -> Tuple[int, int, int]:
        """Get shape of activations in last stage (H, W, C)."""
        if self.input_size == 32:
            return (8, 8, 64)
        elif self.input_size == 96:
            return (24, 24, 64)
        else:
            return (self.input_size // 4, self.input_size // 4, 64)


def resnet20_v1(
    num_classes: int = 10,
    in_channels: int = 3,
    input_size: int = 32,
    bn_momentum: float = 0.1,
    bn_eps: float = 1e-5,
    use_bn: bool = True,
    relu_scale: float = 1.0,
    relu_clip: Optional[float] = None,
    residual_scale: float = 1.0,
    shortcut_mode: str = "projection",
) -> ResNetV1:
    """
    ResNet-20 v1 for CIFAR-10/MNIST.

    Structure:
        - depth = 6*3 + 2 = 20
        - 3 stages with 3 blocks each
        - Total activations: 1 + 6*3 = 19
        - Last 31%: 6 activations (8x8x64 each) = 24,576 neurons

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        input_size: Input image size
        bn_momentum: BatchNorm momentum (0.1=PyTorch, 0.01=Keras-style)
        bn_eps: BatchNorm epsilon (1e-5=PyTorch, 1e-3=Keras-style)
        use_bn: Whether to use BatchNorm layers (default True)
    """
    return ResNetV1(
        num_blocks=[3, 3, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        input_size=input_size,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        use_bn=use_bn,
        relu_scale=relu_scale,
        relu_clip=relu_clip,
        residual_scale=residual_scale,
        shortcut_mode=shortcut_mode,
    )


