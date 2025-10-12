# train/cpc.py
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

class CPCDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, patch_size=16):
        self.base_dataset = base_dataset
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
        
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(c, n_patches_h * n_patches_w, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)
        
        return patches

def pretrain(backbone, train_loader, args):
    context_network = nn.GRU(args.feature_dim, args.hidden_dim, batch_first=True).to(args.device)
    predictor = nn.Linear(args.hidden_dim, args.feature_dim).to(args.device)
    
    optimizer = optim.Adam(list(backbone.parameters()) + 
                          list(context_network.parameters()) + 
                          list(predictor.parameters()), 
                          lr=args.pretrain_lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.pretrain_epochs):
        backbone.train()
        context_network.train()
        predictor.train()
        total_loss = 0
        
        for patches in train_loader:
            patches = patches.to(args.device)
            batch_size, n_patches = patches.shape[:2]
            
            patches = patches.view(batch_size * n_patches, *patches.shape[2:])
            z = backbone(patches)
            z = z.view(batch_size, n_patches, -1)
            
            context_steps = n_patches // 2
            z_context = z[:, :context_steps, :]
            z_future = z[:, context_steps:, :]
            
            c, _ = context_network(z_context)
            c_last = c[:, -1, :]
            
            optimizer.zero_grad()
            
            pred = predictor(c_last)
            
            loss = 0
            for i in range(z_future.shape[1]):
                z_target = z_future[:, i, :]
                
                similarity = F.cosine_similarity(pred.unsqueeze(1), z_target.unsqueeze(1), dim=-1)
                
                negative_samples = z_future[torch.randperm(batch_size)][:, i, :]
                neg_similarity = F.cosine_similarity(pred.unsqueeze(1), negative_samples.unsqueeze(1), dim=-1)
                
                loss += -torch.log(torch.exp(similarity) / (torch.exp(similarity) + torch.exp(neg_similarity).sum())).mean()
            
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
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    backbone = YourBackboneModel().to(args.device)
    pretrain(backbone, pretrain_loader, args)
    downstream(backbone, train_loader, val_loader, args, args.num_classes)
