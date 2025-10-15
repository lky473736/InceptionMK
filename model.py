# model.py
import torch
import torch.nn as nn
from torchinfo import summary

class DSConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DSConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            DSConv1D(in_channels, out_channels, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            DSConv1D(in_channels, out_channels, kernel_size=1),
            DSConv1D(out_channels, out_channels, kernel_size=3)
        )
        self.branch3 = nn.Sequential(
            DSConv1D(in_channels, out_channels, kernel_size=1),
            DSConv1D(out_channels, out_channels, kernel_size=7)
        )
        self.branch4 = DSConv1D(in_channels, out_channels, kernel_size=1)
        
        concat_channels = out_channels * 4
        self.skip = nn.Conv1d(in_channels, concat_channels, kernel_size=1) if in_channels != concat_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        concat = torch.cat([out1, out2, out3, out4], dim=1)
        skip = self.skip(x)
        out = concat + skip
        out = self.activation(out)
        return out

class MultiKernelBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiKernelBlock, self).__init__()
        self.branch1 = DSConv1D(in_channels, out_channels, kernel_size=1)
        self.branch2 = DSConv1D(in_channels, out_channels, kernel_size=3)
        self.branch3 = DSConv1D(in_channels, out_channels, kernel_size=5)
        self.branch4 = DSConv1D(in_channels, out_channels, kernel_size=7)
        
        concat_channels = out_channels * 4
        self.skip = nn.Conv1d(in_channels, concat_channels, kernel_size=1) if in_channels != concat_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        concat = torch.cat([out1, out2, out3, out4], dim=1)
        skip = self.skip(x)
        out = concat + skip
        out = self.activation(out)
        return out

class InceptionMK(nn.Module):
    def __init__(self, input_channels=9, stem_out=64, block_out=32, embedding_dim=128, num_classes=10, num_rotations=4):
        super(InceptionMK, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(stem_out),
            nn.ReLU()
        )
        self.inception = InceptionBlock(stem_out, block_out)
        self.mk_block = MultiKernelBlock(block_out * 4, block_out)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(block_out * 4, embedding_dim)
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

    def forward_features(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.inception(x)
        x = self.mk_block(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        embedding = self.embedding(x)
        return embedding

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.inception(x)
        x = self.mk_block(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        embedding = self.embedding(x)
        activity_logits = self.activity_head(embedding)
        rotation_logits = self.rotation_head(embedding)
        return activity_logits, rotation_logits

if __name__ == '__main__':
    model = InceptionMK()
    example_input = torch.randn(2, 100, 9)
    summary(model, input_data=example_input)
