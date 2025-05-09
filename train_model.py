import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import cv2
from preprocess_cards import extract_features
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PokemonCardDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        self.samples = []
        
        # Load all images and their PSA grades
        for psa_grade in range(1, 11):
            grade_dir = os.path.join(data_dir, f'psa{psa_grade}')
            if os.path.exists(grade_dir):
                for img_name in os.listdir(grade_dir):
                    if not img_name.startswith('thresh_'):  # Skip threshold images
                        img_path = os.path.join(grade_dir, img_name)
                        self.samples.append((img_path, psa_grade - 1))  # Convert to 0-9 scale
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, grade = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=img)
                img = transformed["image"]
            else:
                img = self.transform(img)
        
        return img, grade

class CardGradingModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CardGradingModel, self).__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-8]:  # Unfreeze more layers
            param.requires_grad = False
        
        # Modify the final layer for our grading task
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

def get_transforms(is_training=True):
    if is_training:
        return A.Compose([
            A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=45, translate_percent=0.0625, scale=(0.9, 1.1), p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15, device='cuda'):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 3  # Reduced from 5 to 3
    patience_counter = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print('New best model saved!')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs')
            
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            print(f'Best validation accuracy: {best_val_acc:.2f}%')
            break

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets with different transforms for training and validation
    train_dataset = PokemonCardDataset('aligned_samples', transform=get_transforms(is_training=True), is_training=True)
    val_dataset = PokemonCardDataset('aligned_samples', transform=get_transforms(is_training=False), is_training=False)
    
    # Split into train and validation sets (80-20 split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    _, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize model
    model = CardGradingModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15, device=device)

if __name__ == '__main__':
    main() 