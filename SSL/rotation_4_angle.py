# SSL/rotation_4_angle.py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import copy
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import InceptionMK

def apply_3d_rotation(data, angles):
    r = R.from_euler('xyz', angles, degrees=True).as_matrix()
    rm = torch.from_numpy(r).float()
    rotated_data = data.clone()
    for i in [0, 3, 6]:
        rotated_data[i:i+3, :] = torch.matmul(rm, data[i:i+3, :])
    return rotated_data

class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.angles = [0, 90, 180, 270]
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data, _ = self.base_dataset[idx]
        label = np.random.randint(0, 4)
        rotated_data = apply_3d_rotation(data, [0, 0, self.angles[label]])
        return rotated_data, label

def pretrain(backbone, train_loader, args):
    pretext_classifier = nn.Linear(args.feature_dim, 4).to(args.device)
    optimizer = optim.Adam(list(backbone.parameters()) + list(pretext_classifier.parameters()), 
                          lr=args.pretrain_lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.pretrain_epochs):
        backbone.train()
        pretext_classifier.train()
        total_loss, correct, total = 0, 0, 0
        
        for data, labels in train_loader:
            data, labels = data.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            features = backbone.forward_features(data)
            outputs = pretext_classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        print(f'Pretrain Epoch {epoch+1}/{args.pretrain_epochs}: Loss: {total_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%')

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
        correct, total = 0, 0
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
        correct, total = 0, 0
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
    correct, total = 0, 0
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
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    backbone = InceptionMK(input_channels=args.input_channels, embedding_dim=args.feature_dim).to(args.device)
    
    # train_ds = ... (Base training dataset)
    # val_ds = ... (Base validation dataset)
    # pretrain_loader = DataLoader(RotationDataset(train_ds), batch_size=args.batch_size, shuffle=True)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    pretrain(backbone, pretrain_loader, args)
    downstream(backbone, train_loader, val_loader, args, args.num_classes)
