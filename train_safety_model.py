#!/usr/bin/env python3
"""
Train a mushroom safety prediction model using the Kaggle dataset.
This model predicts whether a mushroom is edible or poisonous based on features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import argparse
from pathlib import Path
import kagglehub

def download_and_load_dataset():
    """Download and load the Kaggle mushroom dataset"""
    print("🍄 Downloading Kaggle Mushroom Classification Dataset...")
    
    try:
        # Download the dataset
        dataset_path = kagglehub.dataset_download("uciml/mushroom-classification")
        csv_file = os.path.join(dataset_path, 'mushrooms.csv')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")
        
        print(f"✅ Dataset downloaded to: {dataset_path}")
        
        # Load the data
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} mushroom samples with {len(df.columns)} features")
        
        return df, dataset_path
        
    except Exception as e:
        print(f"❌ Error downloading/loading dataset: {e}")
        return None, None

def prepare_features(df):
    """Prepare features for training"""
    print("\n🔧 Preparing features for training...")
    
    # Separate features and target
    X = df.drop('class', axis=1)  # All columns except 'class'
    y = df['class']  # Target: 'e' for edible, 'p' for poisonous
    
    print(f"  - Features: {X.shape[1]} columns")
    print(f"  - Target classes: {y.value_counts().to_dict()}")
    
    # Encode categorical features
    label_encoders = {}
    X_encoded = X.copy()
    
    for column in X.columns:
        if X[column].dtype == 'object':  # Categorical column
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(X[column])
            label_encoders[column] = le
            print(f"  - Encoded '{column}': {len(le.classes_)} unique values")
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    print(f"  - Target encoding: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")
    
    return X_encoded, y_encoded, label_encoders, target_encoder

def train_model(X, y, test_size=0.2, random_state=42):
    """Train the safety prediction model"""
    print(f"\n🚀 Training safety prediction model...")
    print(f"  - Training set: {int(len(X) * (1-test_size))} samples")
    print(f"  - Test set: {int(len(X) * test_size)} samples")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize and train Random Forest
    print("  - Training Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  - Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return rf_model, X_train, X_test, y_train, y_test, y_pred

def evaluate_model(y_test, y_pred, target_encoder):
    """Evaluate model performance"""
    print("\n📊 Model Evaluation:")
    
    # Classification report
    target_names = target_encoder.classes_
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    print("\n🔍 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("Predicted:")
    print("         Edible  Poisonous")
    print(f"Actual Edible     {cm[0,0]:6d}     {cm[0,1]:6d}")
    print(f"Poisonous         {cm[1,0]:6d}     {cm[1,1]:6d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    
    print(f"\n📊 Additional Metrics:")
    print(f"  - Sensitivity (Poisonous detection): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"  - Specificity (Edible detection): {specificity:.4f} ({specificity*100:.2f}%)")
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }

def save_model_and_encoders(model, label_encoders, target_encoder, output_dir="models"):
    """Save the trained model and encoders"""
    print(f"\n💾 Saving model and encoders to '{output_dir}'...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, "mushroom_safety_model.pkl")
    joblib.dump(model, model_path)
    print(f"  ✅ Model saved: {model_path}")
    
    # Save label encoders
    encoders_path = os.path.join(output_dir, "label_encoders.pkl")
    joblib.dump(label_encoders, encoders_path)
    print(f"  ✅ Label encoders saved: {encoders_path}")
    
    # Save target encoder
    target_encoder_path = os.path.join(output_dir, "target_encoder.pkl")
    joblib.dump(target_encoder, target_encoder_path)
    print(f"  ✅ Target encoder saved: {target_encoder_path}")
    
    return output_dir

def create_prediction_script(output_dir):
    """Create a simple prediction script"""
    script_content = '''#!/usr/bin/env python3
"""
Simple script to use the trained mushroom safety model.
"""

import joblib
import pandas as pd
import numpy as np

def load_model():
    """Load the trained model and encoders"""
    model = joblib.load('models/mushroom_safety_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    return model, label_encoders, target_encoder

def predict_safety(features_dict, model, label_encoders, target_encoder):
    """Predict mushroom safety from features"""
    # Create a DataFrame with the features
    df = pd.DataFrame([features_dict])
    
    # Encode categorical features
    for column, encoder in label_encoders.items():
        if column in df.columns:
            # Handle unseen categories
            if df[column].iloc[0] in encoder.classes_:
                df[column] = encoder.transform(df[column])
            else:
                print(f"Warning: Unknown value '{df[column].iloc[0]}' for feature '{column}'")
                return None
    
    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    # Decode prediction
    predicted_class = target_encoder.inverse_transform([prediction])[0]
    confidence = max(probability)
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': dict(zip(target_encoder.classes_, probability))
    }

if __name__ == "__main__":
    # Example usage
    model, label_encoders, target_encoder = load_model()
    
    # Example features (you can modify these)
    sample_features = {
        'cap-shape': 'convex',
        'cap-surface': 'smooth',
        'cap-color': 'brown',
        'bruises': 'bruises',
        'odor': 'none',
        'gill-attachment': 'free',
        'gill-spacing': 'close',
        'gill-size': 'broad',
        'gill-color': 'white',
        'stalk-shape': 'enlarging',
        'stalk-root': 'equal',
        'stalk-surface-above-ring': 'smooth',
        'stalk-surface-below-ring': 'smooth',
        'stalk-color-above-ring': 'white',
        'stalk-color-below-ring': 'white',
        'veil-type': 'partial',
        'veil-color': 'white',
        'ring-number': 'one',
        'ring-type': 'pendant',
        'spore-print-color': 'white',
        'population': 'scattered',
        'habitat': 'grasses'
    }
    
    result = predict_safety(sample_features, model, label_encoders, target_encoder)
    if result:
        print(f"🍄 Prediction: {result['prediction']}")
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print(f"📊 Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  - {class_name}: {prob:.2%}")
'''
    
    script_path = os.path.join(output_dir, "predict_safety.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"  ✅ Prediction script created: {script_path}")

def main():
    """Main training pipeline"""
    print("🍄 Mushroom Safety Prediction Model Training")
    print("=" * 60)
    
    # Download and load dataset
    df, dataset_path = download_and_load_dataset()
    if df is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Prepare features
    X, y, label_encoders, target_encoder = prepare_features(df)
    
    # Train model
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, target_encoder)
    
    # Save everything
    output_dir = save_model_and_encoders(model, label_encoders, target_encoder)
    
    # Create prediction script
    create_prediction_script(output_dir)
    
    print(f"\n🎉 Training complete!")
    print(f"📁 All files saved to: {os.path.abspath(output_dir)}")
    print(f"\n📋 Next steps:")
    print(f"1. Test the model: python {output_dir}/predict_safety.py")
    print(f"2. Integrate with your Flask app")
    print(f"3. Use for real-time safety predictions")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    main()
