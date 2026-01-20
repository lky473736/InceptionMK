import torch
import torch.nn as nn
from torchinfo import summary

class DSConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DSConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        branch_out = out_channels // 4
        self.branch1 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            DSConv1D(in_channels, branch_out, 1)
        )
        self.branch2 = nn.Sequential(
            DSConv1D(in_channels, branch_out, 1),
            DSConv1D(branch_out, branch_out, 3, padding=1)
        )
        self.branch3 = nn.Sequential(
            DSConv1D(in_channels, branch_out, 1),
            DSConv1D(branch_out, branch_out, 5, padding=2)
        )
        self.branch4 = DSConv1D(in_channels, branch_out, 1)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out += identity
        out = self.relu(out)
        return out

class MultiKernelBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiKernelBlock, self).__init__()
        branch_out = out_channels // 4
        self.branch1 = DSConv1D(in_channels, branch_out, 1)
        self.branch2 = DSConv1D(in_channels, branch_out, 3, padding=1)
        self.branch3 = DSConv1D(in_channels, branch_out, 5, padding=2)
        self.branch4 = DSConv1D(in_channels, branch_out, 7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.bn(out)
        out += identity
        out = self.relu(out)
        return out

class InceptionMK(nn.Module):
    def __init__(self, input_channels=9, stem_out=64, inception_out=128, mk_out=128, num_classes=6, num_rotations=4):
        super(InceptionMK, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(stem_out),
            nn.ReLU()
        )
        self.inception = InceptionBlock(stem_out, inception_out)
        self.mk_block = MultiKernelBlock(inception_out, mk_out)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        self.activity_head = nn.Sequential(
            nn.Linear(mk_out, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(mk_out, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_rotations)
        )

    def forward_features(self, x):
        if x.shape[-1] == 9: 
            x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.inception(x)
        x = self.mk_block(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        activity_logits = self.activity_head(features)
        rotation_logits = self.rotation_head(features)
        return activity_logits, rotation_logits

if __name__ == '__main__':
    model = InceptionMK(input_channels=9, num_classes=6, num_rotations=4)
