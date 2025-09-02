#!/usr/bin/env python3
"""
Improved mushroom image classification training with better accuracy.
This will train for more epochs with transfer learning and better optimization.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import time
from pathlib import Path

class MushroomImageDataset(Dataset):
    """Dataset for loading mushroom images from the uploaded dataset"""
    
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.data = []
        self.class_names = []
        self.class_to_idx = {}
        
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        """Prepare the dataset by scanning all species folders"""
        print("🔍 Scanning mushroom image dataset...")
        
        # Get all species folders
        species_folders = [d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d))]
        species_folders.sort()  # Ensure consistent ordering
        
        print(f"📊 Found {len(species_folders)} species folders")
        
        # Create class mapping
        for idx, species in enumerate(species_folders):
            self.class_names.append(species)
            self.class_to_idx[species] = idx
        
        # Collect all image paths with labels
        for species in species_folders:
            species_path = os.path.join(self.data_dir, species)
            image_files = [f for f in os.listdir(species_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"  - {species}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(species_path, img_file)
                self.data.append({
                    'image_path': img_path,
                    'label': self.class_to_idx[species],
                    'species': species
                })
        
        print(f"✅ Total images: {len(self.data)}")
        
        # Shuffle the data
        np.random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Load and return an image"""
        item = self.data[idx]
        
        try:
            # Load image
            image = Image.open(item['image_path']).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, item['label']
            
        except Exception as e:
            print(f"⚠️  Error loading image {item['image_path']}: {e}")
            # Return a placeholder image if loading fails
            placeholder = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                placeholder = self.transform(placeholder)
            return placeholder, item['label']

def create_improved_model(num_classes):
    """Create an improved ResNet-50 model with transfer learning"""
    print(f"🏗️  Creating improved ResNet-50 model for {num_classes} classes...")
    
    # Use pre-trained weights for better initialization
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze early layers for transfer learning
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last few layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Create improved classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Reduced dropout for better training
        nn.Linear(in_features, 2048),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes)
    )
    
    return model

def get_improved_transforms():
    """Get improved image transformations with better augmentation"""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Added vertical flip
        transforms.RandomRotation(degrees=20),  # Increased rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added affine transforms
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch_improved(model, dataloader, criterion, optimizer, device):
    """Improved training for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate_improved(model, dataloader, criterion, device):
    """Improved validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

def save_improved_model(model, class_names, output_dir="models"):
    """Save the improved model and class mapping"""
    print(f"💾 Saving improved model and mapping to '{output_dir}'...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the improved model
    model_path = os.path.join(output_dir, "mushroom_image_classifier_improved.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  ✅ Improved model saved: {model_path}")
    
    # Save class mapping
    mapping_path = os.path.join(output_dir, "class_mapping_improved.json")
    class_mapping = {
        "classes": class_names,
        "num_classes": len(class_names),
        "model_type": "ResNet-50 Improved",
        "input_size": 224,
        "training_data": "Uploaded Mushroom Dataset (Improved)",
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(class_names),
        "improvements": [
            "Transfer learning with ImageNet weights",
            "More training epochs (50)",
            "Better learning rate scheduling",
            "Improved data augmentation",
            "Batch normalization",
            "Gradient clipping"
        ]
    }
    
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"  ✅ Improved class mapping saved: {mapping_path}")
    
    return output_dir

def main():
    """Main improved training pipeline"""
    print("🍄 Improved Mushroom Image Classification Training")
    print("=" * 60)
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Dataset path
    dataset_path = "mushroom_image_dataset/data"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the mushroom image dataset is uploaded and accessible")
        return
    
    # Get improved transforms
    train_transform, val_transform = get_improved_transforms()
    
    # Create full dataset
    print(f"\n📥 Creating dataset...")
    full_dataset = MushroomImageDataset(dataset_path, train_transform)
    
    # Split into train/val (80/20)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders with more workers
    batch_size = 32  # Increased batch size for better training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset split:")
    print(f"  - Total images: {total_size}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Classes: {len(full_dataset.class_names)}")
    print(f"  - Batch size: {batch_size}")
    
    # Create improved model
    model = create_improved_model(len(full_dataset.class_names))
    model = model.to(device)
    
    # Improved loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Better optimizer
    
    # Improved learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training parameters
    num_epochs = 50  # Significantly more epochs
    
    # Training loop
    print(f"\n🚀 Starting improved training for {num_epochs} epochs...")
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch_improved(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_improved(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f"📊 Epoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"🎯 New best validation accuracy: {best_val_acc:.2f}%")
            
            # Save best model
            best_model_path = os.path.join("models", "best_mushroom_classifier_improved.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 Best improved model saved: {best_model_path}")
        
        # Early stopping if no improvement for 10 epochs
        if epoch > 10 and all(val_accs[-10] >= acc for acc in val_accs[-9:]):
            print(f"🛑 Early stopping at epoch {epoch+1} - no improvement for 10 epochs")
            break
    
    # Save final improved model
    output_dir = save_improved_model(model, full_dataset.class_names)
    
    print(f"\n🎉 Improved training complete!")
    print(f"📁 Model saved to: {os.path.abspath(output_dir)}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    print(f"\n📋 Next steps:")
    print(f"1. Test the improved model with sample images")
    print(f"2. Update your Flask app to use the improved model")
    print(f"3. Upload mushroom images to test better classification!")
    print(f"4. The improved model should recognize death caps much better!")

if __name__ == "__main__":
    main()
