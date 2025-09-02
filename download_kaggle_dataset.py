#!/usr/bin/env python3
"""
Download Kaggle Mushroom Dataset

This script downloads the mushroom classification dataset from Kaggle
and organizes it for training.
"""

import kagglehub
import os
import pandas as pd
import shutil
from pathlib import Path

def download_kaggle_dataset():
    """Download the Kaggle mushroom classification dataset"""
    
    print("🍄 Downloading Kaggle Mushroom Classification Dataset...")
    print("=" * 60)
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("uciml/mushroom-classification")
        print(f"✅ Dataset downloaded to: {path}")
        
        # List what we got
        print("\n📁 Dataset contents:")
        for item in os.listdir(path):
            print(f"  - {item}")
        
        return path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def explore_dataset(dataset_path):
    """Explore the downloaded dataset structure"""
    
    print("\n🔍 Exploring dataset structure...")
    
    # Look for CSV files
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    
    if csv_files:
        print(f"📊 Found CSV files: {csv_files}")
        
        # Read the main CSV file
        main_csv = os.path.join(dataset_path, csv_files[0])
        df = pd.read_csv(main_csv)
        
        print(f"\n📈 Dataset Overview:")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        
        # Show class distribution
        if 'class' in df.columns:
            print(f"\n🍄 Mushroom Classes:")
            class_counts = df['class'].value_counts()
            print(f"  - Total classes: {len(class_counts)}")
            print(f"  - Classes: {list(class_counts.index)}")
            
            print(f"\n📊 Class Distribution:")
            for class_name, count in class_counts.head(10).items():
                print(f"  - {class_name}: {count} samples")
        
        return df
    else:
        print("❌ No CSV files found in dataset")
        return None

def create_training_structure(dataset_path, output_dir="kaggle_mushroom_dataset"):
    """Create a training-ready directory structure"""
    
    print(f"\n🏗️  Creating training structure in: {output_dir}")
    
    # Create directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print("✅ Directory structure created")
    
    return output_dir

def main():
    """Main function to download and explore the dataset"""
    
    print("🚀 Starting Kaggle Mushroom Dataset Download")
    print("=" * 60)
    
    # Download dataset
    dataset_path = download_kaggle_dataset()
    if not dataset_path:
        print("❌ Failed to download dataset")
        return
    
    # Explore dataset
    df = explore_dataset(dataset_path)
    if df is None:
        print("❌ Failed to explore dataset")
        return
    
    # Create training structure
    output_dir = create_training_structure(dataset_path)
    
    print("\n🎉 Dataset download and exploration complete!")
    print(f"📁 Dataset location: {os.path.abspath(dataset_path)}")
    print(f"📁 Training structure: {os.path.abspath(output_dir)}")
    
    print("\n📋 Next steps:")
    print("1. Review the dataset structure above")
    print("2. Decide how to organize images for training")
    print("3. Use the training script: python train_model.py --data_dir kaggle_mushroom_dataset")
    
    # Show sample data
    if df is not None:
        print(f"\n📊 Sample data (first 5 rows):")
        print(df.head())
        
        print(f"\n📊 Dataset info:")
        print(df.info())

if __name__ == "__main__":
    main()
