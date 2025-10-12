# train/byol.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import YourBackboneModel

class BYOLDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

def pretrain(backbone, train_loader, args):
    online_encoder = backbone
    projector = nn.Sequential(
        nn.Linear(args.feature_dim, args.projection_dim),
        nn.BatchNorm1d(args.projection_dim),
        nn.ReLU(),
        nn.Linear(args.projection_dim, args.projection_dim)
    ).to(args.device)
    
    predictor = nn.Sequential(
        nn.Linear(args.projection_dim, args.projection_dim),
        nn.BatchNorm1d(args.projection_dim),
        nn.ReLU(),
        nn.Linear(args.projection_dim, args.projection_dim)
    ).to(args.device)
    
    target_encoder = copy.deepcopy(online_encoder)
    target_projector = copy.deepcopy(projector)
    
    for param in target_encoder.parameters():
        param.requires_grad = False
    for param in target_projector.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(list(online_encoder.parameters()) + 
                          list(projector.parameters()) + 
                          list(predictor.parameters()), 
                          lr=args.pretrain_lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.pretrain_epochs):
        online_encoder.train()
        projector.train()
        predictor.train()
        total_loss = 0
        
        for img1, img2 in train_loader:
            img1, img2 = img1.to(args.device), img2.to(args.device)
            
            optimizer.zero_grad()
            
            online_y1 = projector(online_encoder(img1))
            online_y2 = projector(online_encoder(img2))
            online_z1 = predictor(online_y1)
            online_z2 = predictor(online_y2)
            
            with torch.no_grad():
                target_y1 = target_projector(target_encoder(img1))
                target_y2 = target_projector(target_encoder(img2))
            
            loss1 = 2 - 2 * F.cosine_similarity(online_z1, target_y2.detach(), dim=-1).mean()
            loss2 = 2 - 2 * F.cosine_similarity(online_z2, target_y1.detach(), dim=-1).mean()
            loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for online_params, target_params in zip(online_encoder.parameters(), target_encoder.parameters()):
                    target_params.data = args.ema_tau * target_params.data + (1 - args.ema_tau) * online_params.data
                for online_params, target_params in zip(projector.parameters(), target_projector.parameters()):
                    target_params.data = args.ema_tau * target_params.data + (1 - args.ema_tau) * online_params.data
            
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
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--ema_tau', type=float, default=0.996)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    backbone = YourBackboneModel().to(args.device)
    pretrain(backbone, pretrain_loader, args)
    downstream(backbone, train_loader, val_loader, args, args.num_classes)
