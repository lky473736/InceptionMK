import torch
import torch.nn as nn
from torchinfo import summary

# Depthwise Separable Convolution
class DSConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DSConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Inception Block
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            DSConv1D(in_channels, out_channels, kernel_size=1)
        )
        self.branch2 = DSConv1D(in_channels, out_channels, kernel_size=1)
        self.branch3 = DSConv1D(in_channels, out_channels, kernel_size=1)
        self.branch4 = DSConv1D(in_channels, out_channels, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.relu(out)
        return out

# MultiKernel Block
class MultiKernelBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiKernelBlock, self).__init__()
        self.branch1 = DSConv1D(in_channels, out_channels, kernel_size=1)
        self.branch2 = DSConv1D(in_channels, out_channels, kernel_size=3)
        self.branch3 = DSConv1D(in_channels, out_channels, kernel_size=5)
        self.branch4 = DSConv1D(in_channels, out_channels, kernel_size=7)

        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.relu(out)
        return out

# Main
class InceptionMK(nn.Module):
    def __init__(self, input_channels=9, stem_out=64, block_out=32, embedding_dim=128, num_classes=10, num_rotations=4):
        super(InceptionMK, self).__init__()

        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(stem_out),
            nn.ReLU()
        )

        # Inception -> MultiKernel 
        self.inception = InceptionBlock(stem_out, block_out)
        self.mk_block = MultiKernelBlock(block_out * 4, block_out)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.embedding = nn.Linear(block_out * 4, embedding_dim)

        # 2 heads for activity and rotation prediction
        self.activity_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.rotation_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 128),
            nn.Dropout(0.3),
            nn.Linear(128, num_rotations)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T] for Conv1d
        x = self.stem(x)
        x = self.inception(x)
        x = self.mk_block(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        embedding = self.embedding(x)
        activity_logits = self.activity_head(embedding)
        rotation_logits = self.rotation_head(embedding)
        return activity_logits, rotation_logits

model = InceptionMK()

example_input = torch.randn(2, 100, 9) # like WISDM
summary(model, input_data=example_input)
