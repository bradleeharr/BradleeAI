import torchvision as torchvision
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch
from torch.optim.lr_scheduler import StepLR
from utilities import *


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction_ratio=2):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, squeeze_factor=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.se = torchvision.ops.SqueezeExcitation(in_channels, in_channels//squeeze_factor)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        out = self.relu2(out)
        return out


class VariableConvNet(nn.Module):
    def __init__(self, input_channels, num_classes, num_conv_blocks, pooling_interval, hidden_layer_size, dropout, squeeze_factor):
        super(VariableConvNet, self).__init__()
        in_channels = input_channels
        self.dropout = nn.Dropout(dropout)

        # Create an input convolutional layer
        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.relu0 = nn.ReLU()
        conv_layers = []

        # Create a list of convolutional blocks like Maia model
        for i in range(num_conv_blocks):
            conv_layers.append(ResidualBlock(in_channels, squeeze_factor))

        self.conv_layers = nn.ModuleList(conv_layers)

        # Flatten the output and add fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels * 64, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x):
        # Apply the input convolutional layer
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        # Apply the residual convolutional blocks
        for layer in self.conv_layers:
            x = layer(x)
            x = self.dropout(x)

        # Apply the fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class NeuralNetwork(nn.Module):
    def __init__(self, plys, hidden_layer_size, num_conv_blocks, pooling_interval, dropout=0, squeeze_factor=2):
        super(NeuralNetwork, self).__init__()
        # Each board is 8 x 8 board with 12 input channels (6 white pieces, 6 black pieces)
        in_channels = 12 * (plys + 1)
        num_classes = 64 * 64  # From square * To square (4096 possible moves, although not all are valid)

        self.convNet = VariableConvNet(input_channels=in_channels, num_classes=num_classes,
                                       num_conv_blocks=num_conv_blocks,
                                       hidden_layer_size=hidden_layer_size, pooling_interval=pooling_interval,
                                       dropout=dropout,
                                       squeeze_factor=squeeze_factor)

    def forward(self, x):
        x = self.convNet(x)
        return x
