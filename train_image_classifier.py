#!/usr/bin/env python3
"""
Train an image classification model for mushrooms.
This will classify mushroom images into different species and predict safety.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import json
from pathlib import Path
import kagglehub

class MushroomImageDataset(Dataset):
    """Custom dataset for mushroom images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

def download_kaggle_images():
    """Download a Kaggle image dataset for mushrooms"""
    print("🍄 Downloading Kaggle Mushroom Image Dataset...")
    
    # Try to find a good mushroom image dataset
    # For now, we'll use a placeholder approach
    print("⚠️  Note: The current Kaggle dataset is tabular (features only)")
    print("   We need an image dataset for visual classification")
    print("   Let me create a synthetic approach for demonstration...")
    
    return None

def create_synthetic_dataset():
    """Create a synthetic dataset for demonstration purposes"""
    print("🔧 Creating synthetic mushroom image dataset...")
    
    # Create directories
    base_dir = "synthetic_mushroom_dataset"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create class directories
    classes = [
        "chanterelle", "death_cap", "fly_agaric", "porcini", 
        "morel", "false_morel", "destroying_angel", "psilocybe"
    ]
    
    for class_name in classes:
        os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)
    
    # For now, we'll use placeholder images
    # In a real scenario, you'd download actual mushroom images
    print("✅ Synthetic dataset structure created")
    print("📁 Placeholder directories ready for real images")
    
    return base_dir

def create_image_classifier(num_classes=8):
    """Create the image classification model"""
    print(f"🏗️  Creating image classifier for {num_classes} classes...")
    
    # Use pre-trained ResNet-50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Modify the final layer for our number of classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

def get_transforms():
    """Get image transformations for training and validation"""
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

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    """Train the image classification model"""
    print(f"🚀 Training model for {num_epochs} epochs...")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
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
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

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
        "input_size": 224
    }
    
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"  ✅ Class mapping saved: {mapping_path}")
    
    return output_dir

def create_prediction_script(output_dir):
    """Create a script for making predictions with the trained model"""
    script_content = '''#!/usr/bin/env python3
"""
Predict mushroom species from images using the trained model.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

class MushroomImageClassifier:
    def __init__(self, model_path, class_mapping_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.class_mapping = self._load_class_mapping(class_mapping_path)
        self.transform = self._get_transform()
    
    def _load_model(self, model_path):
        """Load the trained model"""
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 8)  # 8 mushroom classes
        )
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def _load_class_mapping(self, mapping_path):
        """Load class mapping"""
        with open(mapping_path, 'r') as f:
            return json.load(f)
    
    def _get_transform(self):
        """Get image transformation"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=3):
        """Predict mushroom species from image"""
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Format results
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            class_name = self.class_mapping['classes'][class_idx]
            confidence = top_probs[0][i].item()
            
            # Add safety information based on class name
            safety_info = self._get_safety_info(class_name)
            
            results.append({
                'class': class_name,
                'confidence': confidence,
                'safety': safety_info['safety'],
                'edible': safety_info['edible'],
                'poisonous': safety_info['poisonous'],
                'psychedelic': safety_info['psychedelic'],
                'warning': safety_info['warning']
            })
        
        return results
    
    def _get_safety_info(self, class_name):
        """Get safety information for a mushroom class"""
        safety_map = {
            'chanterelle': {'safety': 'edible', 'edible': True, 'poisonous': False, 'psychedelic': False, 'warning': '✅ Generally safe to eat when properly identified'},
            'porcini': {'safety': 'edible', 'edible': True, 'poisonous': False, 'psychedelic': False, 'warning': '✅ Generally safe to eat when properly identified'},
            'morel': {'safety': 'edible', 'edible': True, 'poisonous': False, 'psychedelic': False, 'warning': '✅ Generally safe to eat when properly identified'},
            'death_cap': {'safety': 'poisonous', 'edible': False, 'poisonous': True, 'psychedelic': False, 'warning': '☠️ EXTREMELY DEADLY - Causes liver failure and death!'},
            'fly_agaric': {'safety': 'poisonous', 'edible': False, 'poisonous': True, 'psychedelic': True, 'warning': '☠️ EXTREMELY DANGEROUS - Contains toxic compounds!'},
            'false_morel': {'safety': 'poisonous', 'edible': False, 'poisonous': True, 'psychedelic': False, 'warning': '☠️ EXTREMELY DANGEROUS - Contains gyromitrin toxin!'},
            'destroying_angel': {'safety': 'poisonous', 'edible': False, 'poisonous': True, 'psychedelic': False, 'warning': '☠️ EXTREMELY DEADLY - Causes organ failure!'},
            'psilocybe': {'safety': 'psychedelic', 'edible': False, 'poisonous': False, 'psychedelic': True, 'warning': '⚠️ PSYCHOACTIVE - Contains hallucinogenic compounds!'}
        }
        
        return safety_map.get(class_name, {'safety': 'unknown', 'edible': False, 'poisonous': False, 'psychedelic': False, 'warning': '❓ Unknown edibility'})

if __name__ == "__main__":
    # Example usage
    model_path = "models/mushroom_image_classifier.pth"
    mapping_path = "models/class_mapping.json"
    
    if not os.path.exists(model_path) or not os.path.exists(mapping_path):
        print("❌ Model files not found. Please train the model first.")
        sys.exit(1)
    
    classifier = MushroomImageClassifier(model_path, mapping_path)
    
    # Test with a sample image (you'll need to provide an actual image)
    print("🍄 Mushroom Image Classifier Ready!")
    print("📁 Use: classifier.predict('path/to/image.jpg')")
'''
    
    script_path = os.path.join(output_dir, "predict_image.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"  ✅ Prediction script created: {script_path}")

def main():
    """Main training pipeline"""
    print("🍄 Mushroom Image Classification Model Training")
    print("=" * 60)
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # For now, create synthetic dataset structure
    # In a real scenario, you'd download actual mushroom images
    dataset_dir = create_synthetic_dataset()
    
    # Define classes
    class_names = [
        "chanterelle", "death_cap", "fly_agaric", "porcini", 
        "morel", "false_morel", "destroying_angel", "psilocybe"
    ]
    
    print(f"\n📊 Classes: {class_names}")
    print(f"📁 Dataset directory: {dataset_dir}")
    
    # Create model
    model = create_image_classifier(len(class_names))
    
    print(f"\n⚠️  IMPORTANT NOTE:")
    print(f"   This script creates the model structure and training pipeline")
    print(f"   However, you need actual mushroom images to train on!")
    print(f"   The current Kaggle dataset is tabular (features only)")
    print(f"   For real image classification, you need:")
    print(f"   1. A dataset with actual mushroom images")
    print(f"   2. Images organized by species in folders")
    print(f"   3. Sufficient images per class (100+ per species)")
    
    # Save model structure and mapping
    output_dir = save_model_and_mapping(model, class_names)
    
    # Create prediction script
    create_prediction_script(output_dir)
    
    print(f"\n🎉 Setup complete!")
    print(f"📁 Files saved to: {os.path.abspath(output_dir)}")
    print(f"\n📋 Next steps:")
    print(f"1. Find a mushroom image dataset (e.g., iNaturalist, GBIF)")
    print(f"2. Organize images into class folders")
    print(f"3. Update the training script with real data paths")
    print(f"4. Train the model on actual images")
    print(f"5. Integrate with your Flask app for real-time classification")

if __name__ == "__main__":
    main()
