# SSL/masked_reconstruction.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import InceptionMK

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, mask_ratio=0.25):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data, _ = self.base_dataset[idx]
        seq_len = data.shape[0]
        n_mask = int(seq_len * self.mask_ratio)
        mask_indices = torch.randperm(seq_len)[:n_mask]
        
        masked_data = data.clone()
        masked_data[mask_indices] = 0
        
        return masked_data, data

def pretrain(backbone, train_loader, args):
    decoder = nn.Sequential(
        nn.Linear(args.feature_dim, 256),
        nn.ReLU(),
        nn.Linear(256, args.seq_len * args.input_channels)
    ).to(args.device)
    
    optimizer = optim.Adam(list(backbone.parameters()) + list(decoder.parameters()), 
                          lr=args.pretrain_lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    for epoch in range(args.pretrain_epochs):
        backbone.train()
        decoder.train()
        total_loss = 0
        
        for masked_data, original_data in train_loader:
            masked_data = masked_data.to(args.device)
            original_data = original_data.to(args.device)
            
            optimizer.zero_grad()
            features = backbone.forward_features(masked_data)
            reconstructed = decoder(features)
            reconstructed = reconstructed.view(original_data.shape)
            
            loss = criterion(reconstructed, original_data)
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
        
        for data, labels in train_loader:
            data, labels = data.to(args.device), labels.to(args.device)
            with torch.no_grad():
                features = backbone.forward_features(data)
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
        
        for data, labels in train_loader:
            data, labels = data.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            features = backbone.forward_features(data)
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
        for data, labels in val_loader:
            data, labels = data.to(args.device), labels.to(args.device)
            features = backbone.forward_features(data)
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
    parser.add_argument('--input_channels', type=int, default=9)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    backbone = InceptionMK(input_channels=args.input_channels, embedding_dim=args.feature_dim).to(args.device)
    pretrain(backbone, pretrain_loader, args)
    downstream(backbone, train_loader, val_loader, args, args.num_classes)
