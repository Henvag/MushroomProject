#!/usr/bin/env python3
"""
Test the trained mushroom image classification model.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

def load_model():
    """Load the trained model and class mapping"""
    print("🔍 Loading trained mushroom image classifier...")
    
    # Load class mapping
    with open('models/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    
    print(f"📊 Model info:")
    print(f"  - Classes: {class_mapping['num_classes']}")
    print(f"  - Model type: {class_mapping['model_type']}")
    print(f"  - Training data: {class_mapping['training_data']}")
    print(f"  - Training date: {class_mapping['training_date']}")
    
    # Create model architecture
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, class_mapping['num_classes'])
    )
    
    # Load trained weights
    model.load_state_dict(torch.load('models/mushroom_image_classifier.pth', map_location='cpu'))
    model.eval()
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, class_mapping, transform

def predict_image(image_path, model, class_mapping, transform):
    """Predict mushroom species from image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, 5)
        
        # Format results
        results = []
        for i in range(5):
            class_idx = top_indices[0][i].item()
            class_name = class_mapping['classes'][class_idx]
            confidence = top_probs[0][i].item()
            
            results.append({
                'species': class_name.replace('_', ' ').title(),
                'confidence': confidence,
                'percentage': confidence * 100
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def main():
    """Test the model with sample images"""
    print("🍄 Testing Mushroom Image Classification Model")
    print("=" * 60)
    
    # Load model
    model, class_mapping, transform = load_model()
    
    # Test with a few sample images from the dataset
    test_images = [
        "mushroom_image_dataset/data/chanterelle/0.png",
        "mushroom_image_dataset/data/deathcap/0.png",
        "mushroom_image_dataset/data/fly_agaric/0.png"
    ]
    
    print(f"\n🧪 Testing with {len(test_images)} sample images...")
    
    for i, image_path in enumerate(test_images):
        if os.path.exists(image_path):
            print(f"\n📸 Test {i+1}: {os.path.basename(image_path)}")
            print("-" * 40)
            
            results = predict_image(image_path, model, class_mapping, transform)
            
            if results:
                print("🔍 Top 5 predictions:")
                for j, result in enumerate(results):
                    print(f"  {j+1}. {result['species']}: {result['percentage']:.2f}%")
            else:
                print("❌ Prediction failed")
        else:
            print(f"⚠️  Image not found: {image_path}")
    
    print(f"\n✅ Model test complete!")
    print(f"🎯 Your model can now classify {class_mapping['num_classes']} mushroom species!")
    print(f"📱 Ready to integrate with your Flask app!")

if __name__ == "__main__":
    main()
