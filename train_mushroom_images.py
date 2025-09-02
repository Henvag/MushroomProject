#!/usr/bin/env python3
"""
Train mushroom image classification model using the uploaded mushroom image dataset.
This will train on the 216+ species with actual images for real classification.
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

def create_model(num_classes):
    """Create the image classification model"""
    print(f"🏗️  Creating ResNet-50 model for {num_classes} classes...")
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Modify final layer for our number of classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    return model

def get_transforms():
    """Get image transformations"""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
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

def validate(model, dataloader, criterion, device):
    """Validate the model"""
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

def save_model_and_mapping(model, class_names, output_dir="models"):
    """Save the trained model and class mapping"""
    print(f"💾 Saving model and mapping to '{output_dir}'...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, "mushroom_image_classifier.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  ✅ Model saved: {model_path}")
    
    # Save class mapping
    mapping_path = os.path.join(output_dir, "class_mapping.json")
    class_mapping = {
        "classes": class_names,
        "num_classes": len(class_names),
        "model_type": "ResNet-50",
        "input_size": 224,
        "training_data": "Uploaded Mushroom Dataset",
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(class_names)
    }
    
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"  ✅ Class mapping saved: {mapping_path}")
    
    return output_dir

def main():
    """Main training pipeline"""
    print("🍄 Mushroom Image Classification Training")
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
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
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
    
    # Create dataloaders
    batch_size = 32  # Adjust based on your GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset split:")
    print(f"  - Total images: {total_size}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Classes: {len(full_dataset.class_names)}")
    print(f"  - Batch size: {batch_size}")
    
    # Create model
    model = create_model(len(full_dataset.class_names))
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training parameters
    num_epochs = 15  # More epochs for better accuracy
    
    # Training loop
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
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
            best_model_path = os.path.join("models", "best_mushroom_classifier.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 Best model saved: {best_model_path}")
    
    # Save final model
    output_dir = save_model_and_mapping(model, full_dataset.class_names)
    
    print(f"\n🎉 Training complete!")
    print(f"📁 Model saved to: {os.path.abspath(output_dir)}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    print(f"\n📋 Next steps:")
    print(f"1. Test the model with sample images")
    print(f"2. Restart your Flask app to load the new model")
    print(f"3. Upload mushroom images to test real classification!")
    print(f"4. The model can now classify {len(full_dataset.class_names)} different mushroom species!")

if __name__ == "__main__":
    main()
