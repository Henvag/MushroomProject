from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import random
import joblib
import pandas as pd
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Load mushroom database
def load_mushroom_data():
    try:
        with open('mushroom_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_mushroom_data(data):
    with open('mushroom_data.json', 'w') as f:
        json.dump(data, f, indent=2)

# Load ML models if available
def load_ml_models():
    """Load both safety prediction and image classification models"""
    models = {}
    
    # Load safety prediction model
    try:
        safety_model_path = 'models/mushroom_safety_model.pkl'
        encoders_path = 'models/label_encoders.pkl'
        target_encoder_path = 'models/target_encoder.pkl'
        
        if all(os.path.exists(p) for p in [safety_model_path, encoders_path, target_encoder_path]):
            models['safety_model'] = joblib.load(safety_model_path)
            models['label_encoders'] = joblib.load(encoders_path)
            models['target_encoder'] = joblib.load(target_encoder_path)
            print("✅ Safety prediction model loaded successfully!")
        else:
            print("⚠️  Safety prediction model files not found")
            models['safety_model'] = None
    except Exception as e:
        print(f"❌ Error loading safety model: {e}")
        models['safety_model'] = None
    
    # Load image classification model
    try:
        image_model_path = 'models/mushroom_image_classifier.pth'
        class_mapping_path = 'models/class_mapping.json'
        
        if all(os.path.exists(p) for p in [image_model_path, class_mapping_path]):
            # Import PyTorch here to avoid import errors if not installed
            import torch
            import torch.nn as nn
            from torchvision import transforms, models as torch_models
            
            # Load class mapping
            with open(class_mapping_path, 'r') as f:
                models['class_mapping'] = json.load(f)
            
            # Create model architecture (must match exactly what was trained)
            model = torch_models.resnet50(pretrained=False)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, models['class_mapping']['num_classes'])
            )
            
            # Load trained weights
            model.load_state_dict(torch.load(image_model_path, map_location='cpu'))
            model.eval()
            models['image_model'] = model
            models['image_transform'] = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("✅ Image classification model loaded successfully!")
        else:
            print("⚠️  Image classification model files not found")
            models['image_model'] = None
    except Exception as e:
        print(f"❌ Error loading image classification model: {e}")
        models['image_model'] = None
    
    return models

# Initialize with sample data if empty
def init_sample_data():
    mushrooms = load_mushroom_data()
    if not mushrooms:
        sample_mushrooms = [
            {
                "id": 1,
                "name": "Chanterelle",
                "scientific_name": "Cantharellus cibarius",
                "edible": True,
                "poisonous": False,
                "psychedelic": False,
                "taste": "Delicate, slightly peppery with fruity apricot aroma",
                "habitat": "Forest floors, often near oak and pine trees",
                "season": "Summer to Fall",
                "confidence": 0.85,
                "image_url": "/images/chanterelle.jpg"
            },
            {
                "id": 2,
                "name": "Death Cap",
                "scientific_name": "Amanita phalloides",
                "edible": False,
                "poisonous": True,
                "psychedelic": False,
                "taste": "DO NOT EAT - Extremely toxic",
                "habitat": "Under oak and beech trees",
                "season": "Summer to Fall",
                "confidence": 0.90,
                "image_url": "/images/death-cap.jpg"
            },
            {
                "id": 3,
                "name": "Fly Agaric",
                "scientific_name": "Amanita muscaria",
                "edible": False,
                "poisonous": True,
                "psychedelic": True,
                "taste": "DO NOT EAT - Contains muscimol and ibotenic acid",
                "habitat": "Under birch, pine, and spruce trees",
                "season": "Summer to Fall",
                "confidence": 0.92,
                "image_url": "/images/fly-agaric.jpg"
            },
            {
                "id": 4,
                "name": "Psilocybe Cubensis",
                "scientific_name": "Psilocybe cubensis",
                "edible": False,
                "poisonous": False,
                "psychedelic": True,
                "taste": "Not recommended for consumption",
                "habitat": "Grasslands, pastures, cow dung",
                "season": "Spring to Fall",
                "confidence": 0.75,
                "image_url": "/images/psilocybe-cubensis.jpg"
            },
            {
                "id": 5,
                "name": "Porcini",
                "scientific_name": "Boletus edulis",
                "edible": True,
                "poisonous": False,
                "psychedelic": False,
                "taste": "Rich, nutty, and meaty flavor",
                "habitat": "Coniferous and deciduous forests",
                "season": "Summer to Fall",
                "confidence": 0.88,
                "image_url": "/images/porcini.jpg"
            },
            {
                "id": 6,
                "name": "Destroying Angel",
                "scientific_name": "Amanita bisporigera",
                "edible": False,
                "poisonous": True,
                "psychedelic": False,
                "taste": "DO NOT EAT - Extremely deadly",
                "habitat": "Forests, often near oak trees",
                "season": "Summer to Fall",
                "confidence": 0.95,
                "image_url": "/images/destroying-angel.jpg"
            },
            {
                "id": 7,
                "name": "Morel",
                "scientific_name": "Morchella esculenta",
                "edible": True,
                "poisonous": False,
                "psychedelic": False,
                "taste": "Earthy, nutty, and meaty flavor",
                "habitat": "Forests, especially after fires",
                "season": "Spring",
                "confidence": 0.82,
                "image_url": "/images/morel.jpg"
            },
            {
                "id": 8,
                "name": "False Morel",
                "scientific_name": "Gyromitra esculenta",
                "edible": False,
                "poisonous": True,
                "psychedelic": False,
                "taste": "DO NOT EAT - Contains gyromitrin toxin",
                "habitat": "Coniferous forests",
                "season": "Spring",
                "confidence": 0.87,
                "image_url": "/images/false-morel.jpg"
            }
        ]
        save_mushroom_data(sample_mushrooms)

# Load the ML models on startup
ml_models = load_ml_models()

@app.route('/api/mushrooms', methods=['GET'])
def get_mushrooms():
    """Get all mushrooms or filter by properties"""
    mushrooms = load_mushroom_data()
    
    # Filter parameters
    edible = request.args.get('edible')
    poisonous = request.args.get('poisonous')
    psychedelic = request.args.get('psychedelic')
    
    if edible is not None:
        mushrooms = [m for m in mushrooms if m['edible'] == (edible.lower() == 'true')]
    if poisonous is not None:
        mushrooms = [m for m in mushrooms if m['poisonous'] == (poisonous.lower() == 'true')]
    if psychedelic is not None:
        mushrooms = [m for m in mushrooms if m['psychedelic'] == (psychedelic.lower() == 'true')]
    
    return jsonify(mushrooms)

@app.route('/api/mushrooms/<int:mushroom_id>', methods=['GET'])
def get_mushroom(mushroom_id):
    """Get specific mushroom by ID"""
    mushrooms = load_mushroom_data()
    mushroom = next((m for m in mushrooms if m['id'] == mushroom_id), None)
    
    if mushroom:
        return jsonify(mushroom)
    else:
        return jsonify({"error": "Mushroom not found"}), 404

@app.route('/api/identify', methods=['POST'])
def identify_mushroom():
    """Identify mushroom from uploaded image using ML model"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        
        # Check if we have the image classification model
        if ml_models.get('image_model') and ml_models.get('class_mapping'):
            try:
                # Process the image with the ML model
                image = Image.open(image_file.stream).convert('RGB')
                
                # Transform image for model input
                image_tensor = ml_models['image_transform'](image).unsqueeze(0)
                
                # Make prediction
                import torch
                with torch.no_grad():
                    outputs = ml_models['image_model'](image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    top_probs, top_indices = torch.topk(probabilities, 3)
                
                # Format results
                results = []
                for i in range(3):
                    class_idx = top_indices[0][i].item()
                    class_name = ml_models['class_mapping']['classes'][class_idx]
                    confidence = top_probs[0][i].item()
                    
                    # Get safety information
                    safety_info = get_safety_info_for_class(class_name)
                    
                    # Find matching mushroom in database
                    mushroom_data = find_mushroom_by_class(class_name)
                    
                    # Convert uploaded image to base64 for display
                    import base64
                    image_buffer = io.BytesIO()
                    image.save(image_buffer, format='PNG')
                    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
                    uploaded_image_data = f"data:image/png;base64,{image_base64}"
                    
                    if mushroom_data:
                        result = {
                            **mushroom_data,
                            'confidence': confidence,
                            'ml_class': class_name,
                            'uploaded_image': uploaded_image_data
                        }
                    else:
                        result = {
                            'id': i + 1,
                            'name': class_name.replace('_', ' ').title(),
                            'scientific_name': f"Unknown {class_name}",
                            'edible': safety_info['edible'],
                            'poisonous': safety_info['poisonous'],
                            'psychedelic': safety_info['psychedelic'],
                            'taste': safety_info['warning'],
                            'habitat': 'Unknown',
                            'season': 'Unknown',
                            'confidence': confidence,
                            'uploaded_image': uploaded_image_data,
                            'ml_class': class_name
                        }
                    
                    results.append(result)
                
                return jsonify({
                    "matches": results,
                    "message": "ML image identification complete - Always verify with experts before consumption!",
                    "method": "ML Image Classification"
                })
                
            except Exception as e:
                print(f"ML image classification failed: {e}")
                # Fall back to simulation
                return simulate_identification()
        else:
            # Fall back to simulation if ML model not available
            return simulate_identification()
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def simulate_identification():
    """Fallback simulated identification when ML is not available"""
    mushrooms = load_mushroom_data()
    
    # Simulate image processing delay
    import time
    time.sleep(1)
    
    # Simulate more realistic identification logic
    # Prioritize poisonous mushrooms for safety
    poisonous_mushrooms = [m for m in mushrooms if m['poisonous']]
    edible_mushrooms = [m for m in mushrooms if m['edible'] and not m['poisonous']]
    other_mushrooms = [m for m in mushrooms if not m['edible'] and not m['poisonous']]
    
    # 40% chance to identify a poisonous mushroom (for safety)
    # 35% chance to identify an edible mushroom
    # 25% chance to identify other types
    rand = random.random()
    
    if rand < 0.4 and poisonous_mushrooms:
        # Identify a poisonous mushroom
        selected_mushrooms = random.sample(poisonous_mushrooms, min(2, len(poisonous_mushrooms)))
        # Add some edible look-alikes for comparison
        if edible_mushrooms:
            selected_mushrooms.extend(random.sample(edible_mushrooms, min(1, len(edible_mushrooms))))
    elif rand < 0.75 and edible_mushrooms:
        # Identify an edible mushroom
        selected_mushrooms = random.sample(edible_mushrooms, min(2, len(edible_mushrooms)))
        # Add some poisonous look-alikes for safety
        if poisonous_mushrooms:
            selected_mushrooms.extend(random.sample(poisonous_mushrooms, min(1, len(poisonous_mushrooms))))
    else:
        # Identify other types
        selected_mushrooms = random.sample(other_mushrooms, min(2, len(other_mushrooms)))
        # Add variety
        if mushrooms:
            remaining = [m for m in mushrooms if m not in selected_mushrooms]
            if remaining:
                selected_mushrooms.extend(random.sample(remaining, min(1, len(remaining))))
    
    # Ensure we have at least 2-3 results
    if len(selected_mushrooms) < 2:
        remaining = [m for m in mushrooms if m not in selected_mushrooms]
        selected_mushrooms.extend(random.sample(remaining, min(2 - len(selected_mushrooms), len(remaining))))
    
    # Add realistic confidence scores
    for match in selected_mushrooms:
        # Poisonous mushrooms get higher confidence (safety first)
        if match['poisonous']:
            match['confidence'] = round(random.uniform(0.75, 0.95), 2)
        else:
            match['confidence'] = round(random.uniform(0.65, 0.90), 2)
    
    # Sort by confidence
    selected_mushrooms.sort(key=lambda x: x['confidence'], reverse=True)
    
    return jsonify({
        "matches": selected_mushrooms,
        "message": "Simulated identification complete - Always verify with experts before consumption!",
        "method": "Simulation"
    })

def get_safety_info_for_class(class_name):
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

def find_mushroom_by_class(class_name):
    """Find mushroom data by ML class name"""
    mushrooms = load_mushroom_data()
    
    # Map ML class names to database names
    class_mapping = {
        'chanterelle': 'Chanterelle',
        'death_cap': 'Death Cap',
        'fly_agaric': 'Fly Agaric',
        'porcini': 'Porcini',
        'morel': 'Morel',
        'false_morel': 'False Morel',
        'destroying_angel': 'Destroying Angel',
        'psilocybe': 'Psilocybe Cubensis'
    }
    
    database_name = class_mapping.get(class_name)
    if database_name:
        return next((m for m in mushrooms if m['name'] == database_name), None)
    
    return None

@app.route('/api/search', methods=['GET'])
def search_mushrooms():
    """Search mushrooms by name"""
    query = request.args.get('q', '').lower()
    mushrooms = load_mushroom_data()
    
    if query:
        results = [
            m for m in mushrooms 
            if query in m['name'].lower() or query in m['scientific_name'].lower()
        ]
    else:
        results = mushrooms
    
    return jsonify(results)

@app.route('/api/check-safety', methods=['POST'])
def check_safety():
    """Check if mushroom is safe to consume"""
    data = request.get_json()
    mushroom_id = data.get('mushroom_id')
    
    mushrooms = load_mushroom_data()
    mushroom = next((m for m in mushrooms if m['id'] == mushroom_id), None)
    
    if not mushroom:
        return jsonify({"error": "Mushroom not found"}), 404
    
    safety_info = {
        "safe_to_eat": mushroom['edible'] and not mushroom['poisonous'],
        "poisonous": mushroom['poisonous'],
        "psychedelic": mushroom['psychedelic'],
        "warning": ""
    }
    
    if mushroom['poisonous']:
        if mushroom['name'] == 'Fly Agaric':
            safety_info["warning"] = "☠️ EXTREMELY DANGEROUS - Fly Agaric contains muscimol and ibotenic acid. Can cause severe poisoning, hallucinations, and death!"
        elif mushroom['name'] == 'Death Cap':
            safety_info["warning"] = "☠️ EXTREMELY DEADLY - Death Cap causes liver failure and death. No known antidote!"
        elif mushroom['name'] == 'Destroying Angel':
            safety_info["warning"] = "☠️ EXTREMELY DEADLY - Destroying Angel causes organ failure and death within days!"
        elif mushroom['name'] == 'False Morel':
            safety_info["warning"] = "☠️ EXTREMELY DANGEROUS - False Morel contains gyromitrin toxin. Can cause severe poisoning and death!"
        else:
            safety_info["warning"] = "⚠️ POISONOUS - Do not consume! Can cause severe illness or death."
    elif mushroom['psychedelic']:
        safety_info["warning"] = "⚠️ PSYCHOACTIVE - Contains hallucinogenic compounds. Not recommended for consumption."
    elif mushroom['edible']:
        safety_info["warning"] = "✅ Generally safe to eat when properly identified by experts"
    else:
        safety_info["warning"] = "❓ Unknown edibility - Always consult with mycologists before consumption"
    
    return jsonify(safety_info)

@app.route('/api/predict-safety', methods=['POST'])
def predict_safety():
    """Predict mushroom safety using the trained ML model"""
    if not all([ml_models.get('safety_model'), ml_models.get('label_encoders'), ml_models.get('target_encoder')]):
        return jsonify({
            "error": "Safety prediction model not available",
            "message": "Please ensure the model is trained and available"
        }), 503
    
    try:
        data = request.get_json()
        features = data.get('features', {})
        
        if not features:
            return jsonify({"error": "No features provided"}), 400
        
        # Create DataFrame with features
        df = pd.DataFrame([features])
        
        # Encode categorical features
        for column, encoder in ml_models['label_encoders'].items():
            if column in df.columns:
                if df[column].iloc[0] in encoder.classes_:
                    df[column] = encoder.transform(df[column])
                else:
                    return jsonify({
                        "error": f"Invalid value '{df[column].iloc[0]}' for feature '{column}'",
                        "valid_values": list(encoder.classes_)
                    }), 400
        
        # Make prediction
        prediction = ml_models['safety_model'].predict(df)[0]
        probabilities = ml_models['safety_model'].predict_proba(df)[0]
        
        # Decode prediction
        predicted_class = ml_models['target_encoder'].inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        # Get class probabilities
        class_probs = dict(zip(ml_models['target_encoder'].classes_, probabilities))
        
        result = {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "probabilities": class_probs,
            "safety": "edible" if predicted_class == 'e' else "poisonous",
            "warning": ""
        }
        
        # Add safety warnings
        if predicted_class == 'p':
            result["warning"] = "⚠️ POISONOUS - DO NOT CONSUME! Can cause severe illness or death!"
        else:
            result["warning"] = "✅ EDIBLE - However, ALWAYS verify with experts before consumption!"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/api/kaggle-status', methods=['GET'])
def get_kaggle_status():
    """Get the status of the Kaggle dataset and ML models"""
    safety_model_available = all([ml_models.get('safety_model'), ml_models.get('label_encoders'), ml_models.get('target_encoder')])
    image_model_available = all([ml_models.get('image_model'), ml_models.get('class_mapping')])
    
    return jsonify({
        "dataset_available": True,
        "dataset_type": "Feature-based classification",
        "total_samples": 8124,
        "classes": ["edible", "poisonous"],
        "features": 22,
        "safety_model_available": safety_model_available,
        "image_model_available": image_model_available,
        "message": "Kaggle dataset ready" if safety_model_available else "Dataset available but models not loaded"
    })

@app.route('/api/feature-codes', methods=['GET'])
def get_feature_codes():
    """Get the feature code mappings for the Kaggle dataset"""
    feature_codes = {
        "cap-shape": {
            "x": "convex", "b": "bell", "s": "sunken", "f": "flat", "k": "knobbed", "c": "conical"
        },
        "cap-surface": {
            "s": "smooth", "y": "scaly", "f": "fibrous", "g": "grooves"
        },
        "cap-color": {
            "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", 
            "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"
        },
        "bruises": {
            "t": "bruises", "f": "no"
        },
        "odor": {
            "n": "none", "a": "almond", "l": "anise", "c": "creosote", "y": "fishy", 
            "f": "foul", "m": "musty", "p": "pungent", "s": "spicy"
        },
        "gill-attachment": {
            "f": "free", "a": "attached"
        },
        "gill-spacing": {
            "c": "close", "w": "crowded"
        },
        "gill-size": {
            "b": "broad", "n": "narrow"
        },
        "gill-color": {
            "w": "white", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", 
            "r": "green", "o": "orange", "p": "pink", "e": "red", "u": "purple", 
            "k": "black", "y": "yellow"
        },
        "stalk-shape": {
            "e": "enlarging", "t": "tapering"
        },
        "stalk-root": {
            "e": "equal", "c": "club", "b": "bulbous", "r": "rooted", "?": "missing"
        },
        "stalk-surface-above-ring": {
            "s": "smooth", "f": "fibrous", "y": "scaly", "k": "silky"
        },
        "stalk-surface-below-ring": {
            "s": "smooth", "f": "fibrous", "y": "scaly", "k": "silky"
        },
        "stalk-color-above-ring": {
            "w": "white", "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", 
            "o": "orange", "p": "pink", "e": "red", "y": "yellow"
        },
        "stalk-color-below-ring": {
            "w": "white", "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", 
            "o": "orange", "p": "pink", "e": "red", "y": "yellow"
        },
        "veil-type": {
            "p": "partial"
        },
        "veil-color": {
            "w": "white", "n": "brown", "o": "orange", "y": "yellow"
        },
        "ring-number": {
            "o": "one", "t": "two", "n": "none"
        },
        "ring-type": {
            "p": "pendant", "e": "evanescent", "f": "flaring", "l": "large", "n": "none"
        },
        "spore-print-color": {
            "w": "white", "n": "brown", "b": "buff", "h": "chocolate", "r": "green", 
            "o": "orange", "u": "purple", "k": "black", "y": "yellow"
        },
        "population": {
            "s": "scattered", "n": "numerous", "a": "abundant", "v": "several", 
            "y": "clustered", "c": "clustered"
        },
        "habitat": {
            "g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", 
            "u": "urban", "w": "waste", "d": "woods"
        }
    }
    
    return jsonify(feature_codes)

if __name__ == '__main__':
    init_sample_data()
    app.run(debug=True, port=5001)
