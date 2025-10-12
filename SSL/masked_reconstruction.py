# train/masked_reconstruction.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import YourBackboneModel

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, mask_ratio=0.25, patch_size=16):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        c, h, w = img.shape
        n_patches_h = h // self.patch_size
        n_patches_w = w // self.patch_size
        n_patches = n_patches_h * n_patches_w
        
        n_mask = int(n_patches * self.mask_ratio)
        mask_indices = torch.randperm(n_patches)[:n_mask]
        
        mask = torch.ones(n_patches, dtype=torch.bool)
        mask[mask_indices] = False
        
        masked_img = img.clone()
        for i in range(n_patches):
            row = i // n_patches_w
            col = i % n_patches_w
            if not mask[i]:
                masked_img[:, 
                          row*self.patch_size:(row+1)*self.patch_size,
                          col*self.patch_size:(col+1)*self.patch_size] = 0
        
        return masked_img, img, mask

def pretrain(backbone, train_loader, args):
    decoder = nn.Sequential(
        nn.ConvTranspose2d(args.feature_dim, 256, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 3, 3, 1, 1)
    ).to(args.device)
    
    optimizer = optim.Adam(list(backbone.parameters()) + list(decoder.parameters()), 
                          lr=args.pretrain_lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    for epoch in range(args.pretrain_epochs):
        backbone.train()
        decoder.train()
        total_loss = 0
        
        for masked_img, original_img, mask in train_loader:
            masked_img = masked_img.to(args.device)
            original_img = original_img.to(args.device)
            
            optimizer.zero_grad()
            
            features = backbone(masked_img)
            if len(features.shape) == 2:
                features = features.unsqueeze(-1).unsqueeze(-1)
            
            reconstructed = decoder(features)
            
            if reconstructed.shape != original_img.shape:
                reconstructed = F.interpolate(reconstructed, size=original_img.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(reconstructed, original_img)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Pretrain Epoch {epoch+1}/{args.pretrain_epochs}: Loss: {total_loss/len(train_loader):.4f}')

def downstream(backbone, train_loader, val_loader, args, num_classes):
    classifier = nn.Linear(args.feature_dim, num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()
    
    for param in backbone.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=args.weight_decay)
    phase1_epochs = args.downstream_epochs // 2
    
    for epoch in range(phase1_epochs):
        classifier.train()
        backbone.eval()
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            with torch.no_grad():
                features = backbone(images)
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_acc = evaluate(backbone, classifier, val_loader, args)
        print(f'Phase1 Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, Val Acc: {val_acc:.2f}%')
    
    for param in backbone.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), 
                          lr=0.0001, weight_decay=args.weight_decay)
    phase2_epochs = args.downstream_epochs - phase1_epochs
    
    for epoch in range(phase2_epochs):
        backbone.train()
        classifier.train()
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            features = backbone(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_acc = evaluate(backbone, classifier, val_loader, args)
        print(f'Phase2 Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, Val Acc: {val_acc:.2f}%')

def evaluate(backbone, classifier, val_loader, args):
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            features = backbone(images)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--downstream_epochs', type=int, default=50)
    parser.add_argument('--pretrain_lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    backbone = YourBackboneModel().to(args.device)
    pretrain(backbone, pretrain_loader, args)
    downstream(backbone, train_loader, val_loader, args, args.num_classes)
