#!/usr/bin/env python3
"""
Classical Baseline Model for Architecture Classification
Pure CNN without quantum layer - for comparison with hybrid quantum model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set device to XPU (Intel GPU)
if torch.xpu.is_available():
    device = torch.device('xpu')
    print("✓ XPU (Intel GPU) is available!")
else:
    device = torch.device('cpu')
    print("⚠ XPU not available, using CPU")

print(f"Device set to: {device}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

SELECTED_CLASSES = ['dome(inner)', 'dome(outer)', 'gargoyle', 'stained_glass']
DATA_DIR = '/home/advik/Quantum/Mini Project/architecture_dataset_32x32'
BATCH_SIZE = 8  # Match quantum implementation
N_EPOCHS = 10
LEARNING_RATE = 0.001

# ============================================================================
# DATASET
# ============================================================================

class ArchitectureDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load images and labels
train_images = []
train_labels = []

train_dir = os.path.join(DATA_DIR, 'train')

for class_idx, class_name in enumerate(SELECTED_CLASSES):
    class_path = os.path.join(train_dir, class_name)
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path).convert('RGB')
        train_images.append(img)
        train_labels.append(class_idx)

print(f"Total images loaded: {len(train_images)}")
print(f"Classes: {SELECTED_CLASSES}")
print(f"Images per class: {[train_labels.count(i) for i in range(len(SELECTED_CLASSES))]}")

# Split train into train and validation (80/20)
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

print(f"\nTrain set: {len(train_imgs)} images")
print(f"Validation set: {len(val_imgs)} images")

# Create datasets
train_dataset = ArchitectureDataset(train_imgs, train_lbls, transform=transform)
val_dataset = ArchitectureDataset(val_imgs, val_lbls, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# ============================================================================
# MODEL - PURE CLASSICAL CNN
# ============================================================================

class ClassicalCNN(nn.Module):
    def __init__(self, n_classes=4):
        super(ClassicalCNN, self).__init__()
        
        # Same CNN feature extractor as hybrid model
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Classical classifier (replaces quantum layer + classifier)
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Classify
        output = self.classifier(features)
        
        return output

# Create model
model = ClassicalCNN(n_classes=len(SELECTED_CLASSES))
model.to(device)
print("\n" + "="*70)
print("CLASSICAL BASELINE MODEL")
print("="*70)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Model device: {next(model.parameters()).device}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

# ============================================================================
# TRAINING LOOP
# ============================================================================

train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("\n" + "="*70)
print(f"STARTING TRAINING - {N_EPOCHS} EPOCHS")
print("="*70 + "\n")

# Enable interactive mode for live plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for epoch in range(N_EPOCHS):
    print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
    print("-" * 70)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print("=" * 70)
    
    # Update plots
    ax1.clear()
    ax2.clear()
    
    # Plot loss
    ax1.plot(range(1, epoch+2), train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(range(1, epoch+2), val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Validation Loss (Classical)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(range(1, epoch+2), train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=6)
    ax2.plot(range(1, epoch+2), val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training and Validation Accuracy (Classical)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.1)

# ============================================================================
# FINAL RESULTS
# ============================================================================

print("\n" + "="*70)
print("🎯 FINAL RESULTS - CLASSICAL BASELINE")
print("="*70)
print(f"Best Val Accuracy: {max(val_accs):.2f}%")
print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
print(f"Final Val Accuracy: {val_accs[-1]:.2f}%")
print("="*70 + "\n")

# Save final plot
plt.savefig('classical_baseline_results.png', dpi=150, bbox_inches='tight')
print("📈 Plot saved to: classical_baseline_results.png")

# Save model
torch.save(model.state_dict(), 'classical_baseline_model.pth')
print("💾 Model saved to: classical_baseline_model.pth")

plt.ioff()
plt.show()

print("\n✅ Training complete!")
