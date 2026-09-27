import torch
import torch.nn as nn
import torch.nn.functional as F

from _networks import register_network
from _networks._utils import BaseNetwork


# -------------------------
# Basic Residual Block
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection
        out = out + identity
        out = F.relu(out)

        return out


# -------------------------
# ResNet18 for Class-IL
# -------------------------
@register_network("resnet18_classil")
class ResNet18_classil(BaseNetwork):

    def __init__(
        self,
        num_tasks=5,
        num_classes_per_task=2
    ):
        super(ResNet18_classil, self).__init__()

        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task

        # Total number of classes
        #
        # CIFAR-10 Class-IL:
        # 5 tasks × 2 classes = 10 classes
        #
        self.num_classes = (
            num_tasks * num_classes_per_task
        )

        # -------------------------
        # Initial layer
        # -------------------------
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        # -------------------------
        # ResNet layers
        # -------------------------
        self.layer1 = self._make_layer(
            BasicBlock,
            64,
            2,
            stride=1
        )

        self.layer2 = self._make_layer(
            BasicBlock,
            128,
            2,
            stride=2
        )

        self.layer3 = self._make_layer(
            BasicBlock,
            256,
            2,
            stride=2
        )

        self.layer4 = self._make_layer(
            BasicBlock,
            512,
            2,
            stride=2
        )

        # -------------------------
        # Global Average Pooling
        # -------------------------
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # -------------------------
        # Single classifier
        # -------------------------
        #
        # Class-IL uses ONE classifier
        # for ALL classes.
        #
        # CIFAR-10:
        #     10 output neurons
        #
        self.fc = nn.Linear(
            512 * BasicBlock.expansion,
            self.num_classes
        )

    # -------------------------
    # Make ResNet Layer
    # -------------------------
    def _make_layer(
        self,
        block,
        out_channels,
        blocks,
        stride=1
    ):

        downsample = None

        # Create projection shortcut when
        # dimensions change.
        if (
            stride != 1
            or self.in_channels
            != out_channels * block.expansion
        ):

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels * block.expansion
                )
            )

        layers = []

        # First block
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample
            )
        )

        # Update number of input channels
        self.in_channels = (
            out_channels * block.expansion
        )

        # Remaining blocks
        for _ in range(1, blocks):

            layers.append(
                block(
                    self.in_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)

    # -------------------------
    # Forward
    # -------------------------
    def forward(
        self,
        x,
        task_id=None,
        penultimate=False
    ):

        # -------------------------
        # Initial layer
        # -------------------------
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # -------------------------
        # ResNet backbone
        # -------------------------
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # -------------------------
        # Global average pooling
        # -------------------------
        x = self.avgpool(x)

        # -------------------------
        # Flatten
        # -------------------------
        features = torch.flatten(
            x,
            1
        )

        # -------------------------
        # Single Class-IL classifier
        # -------------------------
        outputs = self.fc(features)

        # -------------------------
        # FedProto compatibility
        # -------------------------
        #
        # fedproto.py calls:
        #
        # network(
        #     inputs,
        #     task_id,
        #     penultimate=True
        # )
        #
        # Therefore return both:
        #
        #   features
        #   outputs
        #
        if penultimate:
            return features, outputs

        return outputs
