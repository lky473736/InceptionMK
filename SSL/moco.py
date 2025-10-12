# train/moco.py
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

class MoCoDataset(torch.utils.data.Dataset):
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
    encoder_q = backbone
    encoder_k = copy.deepcopy(backbone)
    
    for param in encoder_k.parameters():
        param.requires_grad = False
    
    queue = torch.randn(args.projection_dim, args.queue_size).to(args.device)
    queue = F.normalize(queue, dim=0)
    queue_ptr = 0
    
    projection_q = nn.Sequential(
        nn.Linear(args.feature_dim, args.projection_dim),
        nn.ReLU(),
        nn.Linear(args.projection_dim, args.projection_dim)
    ).to(args.device)
    
    projection_k = copy.deepcopy(projection_q)
    for param in projection_k.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(list(encoder_q.parameters()) + list(projection_q.parameters()), 
                          lr=args.pretrain_lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.pretrain_epochs):
        encoder_q.train()
        projection_q.train()
        total_loss = 0
        
        for img1, img2 in train_loader:
            img1, img2 = img1.to(args.device), img2.to(args.device)
            batch_size = img1.size(0)
            
            optimizer.zero_grad()
            
            q = projection_q(encoder_q(img1))
            q = F.normalize(q, dim=1)
            
            with torch.no_grad():
                k = projection_k(encoder_k(img2))
                k = F.normalize(k, dim=1)
            
            l_pos = torch.bmm(q.view(batch_size, 1, -1), k.view(batch_size, -1, 1)).squeeze(-1)
            l_neg = torch.mm(q, queue.clone().detach())
            
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= args.temperature
            labels = torch.zeros(batch_size, dtype=torch.long).to(args.device)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
                    param_k.data = param_k.data * args.momentum + param_q.data * (1. - args.momentum)
                for param_q, param_k in zip(projection_q.parameters(), projection_k.parameters()):
                    param_k.data = param_k.data * args.momentum + param_q.data * (1. - args.momentum)
            
            ptr = queue_ptr
            queue[:, ptr:ptr + batch_size] = k.T
            queue_ptr = (ptr + batch_size) % args.queue_size
            
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
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--queue_size', type=int, default=65536)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    backbone = YourBackboneModel().to(args.device)
    pretrain(backbone, pretrain_loader, args)
    downstream(backbone, train_loader, val_loader, args, args.num_classes)
