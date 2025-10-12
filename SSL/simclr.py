# SSL/simclr.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import InceptionMK

def jitter(x, sigma=0.03):
    return x + torch.randn_like(x) * sigma

def scaling(x, sigma=0.1):
    factor = torch.randn(x.size(0), 1, x.size(2)).to(x.device) * sigma + 1
    return x * factor

def time_warp(x, sigma=0.2, knot=4):
    orig_steps = torch.arange(x.size(1)).float().to(x.device)
    random_warps = torch.randn(knot + 2).to(x.device) * sigma
    warp_steps = torch.linspace(0, x.size(1) - 1, knot + 2).to(x.device)
    warp_steps = warp_steps + random_warps
    warp_steps = torch.clamp(warp_steps, 0, x.size(1) - 1)
    warped = torch.zeros_like(x)
    for i in range(x.size(0)):
        for j in range(x.size(2)):
            warped[i, :, j] = torch.from_numpy(
                np.interp(orig_steps.cpu(), warp_steps.cpu(), x[i, :, j].cpu())
            ).float().to(x.device)
    return warped

def augment(x):
    x = jitter(x)
    x = scaling(x)
    return x

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data, _ = self.base_dataset[idx]
        data1 = augment(data.unsqueeze(0)).squeeze(0)
        data2 = augment(data.unsqueeze(0)).squeeze(0)
        return data1, data2

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    positives = torch.cat([torch.diag(similarity_matrix, batch_size), 
                          torch.diag(similarity_matrix, -batch_size)], dim=0)
    nominator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=1)
    loss = -torch.mean(torch.log(nominator / denominator))
    return loss

def pretrain(backbone, train_loader, args):
    projection_head = nn.Sequential(
        nn.Linear(args.feature_dim, args.projection_dim),
        nn.ReLU(),
        nn.Linear(args.projection_dim, args.projection_dim)
    ).to(args.device)
    
    optimizer = optim.Adam(list(backbone.parameters()) + list(projection_head.parameters()), 
                          lr=args.pretrain_lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.pretrain_epochs):
        backbone.train()
        projection_head.train()
        total_loss = 0
        
        for data1, data2 in train_loader:
            data1, data2 = data1.to(args.device), data2.to(args.device)
            optimizer.zero_grad()
            h1 = backbone.forward_features(data1)
            h2 = backbone.forward_features(data2)
            z1 = projection_head(h1)
            z2 = projection_head(h2)
            loss = nt_xent_loss(z1, z2, args.temperature)
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
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    backbone = InceptionMK(input_channels=args.input_channels, embedding_dim=args.feature_dim).to(args.device)
    pretrain(backbone, pretrain_loader, args)
    downstream(backbone, train_loader, val_loader, args, args.num_classes)
