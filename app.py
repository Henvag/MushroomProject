from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import random
from PIL import Image
import io
import base64
import requests
import sys
import logging

# Configure logging to show all output immediately
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
# Force stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Fix Windows console encoding for emoji characters
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

app = Flask(__name__)
CORS(app)

# Enable Flask debug logging
app.logger.setLevel(logging.DEBUG)
logging.getLogger('werkzeug').setLevel(logging.DEBUG)

# Roboflow configuration - Using Workflow API
# NOTE: Reverted to original hard-coded values per user request
ROBOFLOW_WORKFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY = "n2XmYmT5JN31JxXN6T14"  # Private API key
ROBOFLOW_WORKSPACE = "mushroommodel"  # Workspace name from workflow URL
ROBOFLOW_WORKFLOW_ID = "custom-workflow-2"  # Workflow ID from Unique URL
# Workflow parameters (matching the workflow configuration from schema)
# Note: Field name is "iOU" (capital I, capital O, capital U) - case sensitive!
ROBOFLOW_IOU = 0.5
ROBOFLOW_MAX_DETECTIONS = 50
ROBOFLOW_CONFIDENCE = 0.4

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
                "taste": "DO NOT EAT - Contains gyromitrin toxin",
                "habitat": "Coniferous forests",
                "season": "Spring",
                "confidence": 0.87,
                "image_url": "/images/false-morel.jpg"
            }
        ]
        save_mushroom_data(sample_mushrooms)


@app.route('/api/mushrooms', methods=['GET'])
def get_mushrooms():
    """Get all mushrooms or filter by properties"""
    mushrooms = load_mushroom_data()
    
    # Filter parameters
    edible = request.args.get('edible')
    poisonous = request.args.get('poisonous')
    
    if edible is not None:
        mushrooms = [m for m in mushrooms if m['edible'] == (edible.lower() == 'true')]
    if poisonous is not None:
        mushrooms = [m for m in mushrooms if m['poisonous'] == (poisonous.lower() == 'true')]
    
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
    """Identify mushroom from uploaded image using Roboflow API"""
    try:
        app.logger.info("=" * 60)
        app.logger.info("🔵 IDENTIFY REQUEST RECEIVED")
        app.logger.info("=" * 60)
        sys.stdout.flush()  # Force flush
        
        if 'image' not in request.files:
            app.logger.error("❌ No image file in request")
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        app.logger.info(f"✅ Image file received: {image_file.filename}")
        image = Image.open(image_file.stream).convert('RGB')
        app.logger.info(f"✅ Image opened: {image.size[0]}x{image.size[1]}")
        sys.stdout.flush()
        
        # Use Roboflow workflow inference
        app.logger.info("🚀 Calling roboflow_inference()...")
        sys.stdout.flush()
        roboflow_result = roboflow_inference(image_file)
        
        if roboflow_result:
            app.logger.info(f"✅ Roboflow result received: {type(roboflow_result)}")
            sys.stdout.flush()
            processed_result = process_roboflow_result(roboflow_result, image)
            
            if processed_result:
                # Use the method from processed result (e.g., "Roboflow Workflow AI")
                method = processed_result.get('method', 'Roboflow Workflow AI')
                app.logger.info(f"✅ Returning processed result: method={method}, class={processed_result.get('ml_class')}, confidence={processed_result.get('confidence')}")
                app.logger.info("=" * 60)
                sys.stdout.flush()
                return jsonify({
                    "matches": [processed_result],
                    "message": "Roboflow workflow identification complete - Always verify with experts before consumption!",
                    "method": method
                })
            else:
                app.logger.error("❌ process_roboflow_result returned None")
        else:
            app.logger.error("❌ roboflow_inference returned None")
        
        # Fall back to simulation if Roboflow fails
        app.logger.warning("⚠️ Roboflow workflow failed, falling back to simulation mode")
        if roboflow_result:
            app.logger.info(f"📋 Roboflow result was: {json.dumps(roboflow_result, indent=2)[:1000]}")
        else:
            app.logger.info("📋 Roboflow result was None - API call likely failed")
        app.logger.info("=" * 60)
        sys.stdout.flush()
        return simulate_identification()
        
    except Exception as e:
        app.logger.error(f"❌ EXCEPTION in identify_mushroom: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        sys.stdout.flush()
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


def roboflow_inference(image_file):
    """Perform inference using Roboflow Workflow API"""
    try:
        app.logger.info("🍄 Starting Roboflow workflow inference...")
        app.logger.info(f"📋 API Key: {ROBOFLOW_API_KEY[:10]}...")
        app.logger.info(f"📋 Workspace: {ROBOFLOW_WORKSPACE}")
        app.logger.info(f"📋 Workflow ID: {ROBOFLOW_WORKFLOW_ID}")
        sys.stdout.flush()
        
        # Ensure file pointer is at start, then save the uploaded image temporarily
        try:
            image_file.stream.seek(0)
        except Exception:
            pass
        temp_image_path = 'temp_upload.jpg'
        image_file.save(temp_image_path)
        file_size = os.path.getsize(temp_image_path)
        app.logger.info(f"📁 Image saved to {temp_image_path} ({file_size} bytes)")
        sys.stdout.flush()
        
        # Build workflow API URL
        # Roboflow workflows use: https://serverless.roboflow.com/{workspace}/workflows/{workflow_id}
        workflow_url = f"{ROBOFLOW_WORKFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"
        
        app.logger.info(f"🌐 Calling Roboflow workflow API: {workflow_url}")
        sys.stdout.flush()
        
        # Read image file and convert to base64
        try:
            with open(temp_image_path, 'rb') as img_file:
                image_data = img_file.read()
                # Encode image to base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            app.logger.error(f"❌ Error reading image file: {e}")
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            sys.stdout.flush()
            return None
        
        # Prepare request payload
        # Workflow API expects JSON with api_key and inputs object
        # Based on workflow schema:
        # - "iOU" (case-sensitive! capital I, O, U)
        # - "max_detections"
        # - "confidence"
        # - "image" (InferenceImage type - base64 encoded string)
        inputs = {
            'image': image_base64,
            'iOU': ROBOFLOW_IOU,  # Note: Case-sensitive field name!
            'max_detections': ROBOFLOW_MAX_DETECTIONS,
            'confidence': ROBOFLOW_CONFIDENCE
        }
        
        # Prepare payload with api_key and inputs
        try:
            payload = {
                'api_key': ROBOFLOW_API_KEY,
                'inputs': inputs
            }
            app.logger.info(f"📊 Sending request with workflow parameters:")
            app.logger.info(f"   - image: base64 encoded ({len(image_base64)} chars)")
            app.logger.info(f"   - iOU: {ROBOFLOW_IOU} (note: case-sensitive field name)")
            app.logger.info(f"   - max_detections: {ROBOFLOW_MAX_DETECTIONS}")
            app.logger.info(f"   - confidence: {ROBOFLOW_CONFIDENCE}")
            sys.stdout.flush()
            
            response = requests.post(
                workflow_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            # Log response for debugging
            app.logger.info(f"📡 Response status: {response.status_code}")
            if response.status_code != 200:
                app.logger.warning(f"⚠️ Request failed with status {response.status_code}")
                app.logger.warning(f"📋 Error response: {response.text[:1000]}")
            sys.stdout.flush()
            
            last_error = None
            if response.status_code != 200:
                last_error = f"Status {response.status_code}: {response.text[:500]}"
                
        except Exception as e:
            last_error = str(e)
            app.logger.error(f"❌ Error calling workflow API: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            response = None
            sys.stdout.flush()
        
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            app.logger.info("🗑️ Temporary file cleaned up")
        
        if response and response.status_code == 200:
            try:
                result = response.json()
                app.logger.info("✅ Roboflow workflow inference successful!")
                app.logger.info(f"📋 Response type: {type(result)}")
                if isinstance(result, list):
                    app.logger.info(f"📋 Response is an array with {len(result)} item(s)")
                    if len(result) > 0:
                        app.logger.info(f"📋 First item keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'Not a dict'}")
                elif isinstance(result, dict):
                    app.logger.info(f"📋 Response keys: {list(result.keys())}")
                else:
                    app.logger.info(f"📋 Response: {result}")
                app.logger.info(f"📋 Full response (first 2000 chars): {json.dumps(result, indent=2)[:2000]}")
                sys.stdout.flush()
                return result
            except Exception as e:
                app.logger.error(f"❌ Error parsing JSON response: {e}")
                app.logger.error(f"📋 Response status: {response.status_code}")
                app.logger.error(f"📋 Response text (first 1000 chars): {response.text[:1000]}")
                import traceback
                app.logger.error(traceback.format_exc())
                sys.stdout.flush()
                return None
        else:
            error_msg = last_error or (f"Status {response.status_code}: {response.text[:500]}" if response else "No response")
            app.logger.error(f"❌ Roboflow workflow API error: {error_msg}")
            if response:
                app.logger.error(f"📋 Full error response: {response.text[:2000]}")
            app.logger.error("💡 Tip: Check that the workflow ID, workspace, and API key are correct")
            sys.stdout.flush()
            return None
            
    except requests.exceptions.Timeout:
        app.logger.error("⏰ Roboflow API timeout - request took too long")
        sys.stdout.flush()
        return None
    except requests.exceptions.ConnectionError:
        app.logger.error("🔌 Roboflow API connection error - check internet connection")
        sys.stdout.flush()
        return None
    except Exception as e:
        app.logger.error(f"❌ Roboflow inference error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        sys.stdout.flush()
        return None

def process_roboflow_result(roboflow_result, uploaded_image):
    """Process Roboflow Workflow inference result and format for API response"""
    try:
        app.logger.info(f"🔍 Processing Roboflow result...")
        app.logger.info(f"📋 Result type: {type(roboflow_result)}")
        sys.stdout.flush()
        
        # Convert uploaded image to base64 for display
        image_buffer = io.BytesIO()
        uploaded_image.save(image_buffer, format='PNG')
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        uploaded_image_data = f"data:image/png;base64,{image_base64}"
        
        # Handle workflow output format robustly
        # Roboflow workflow responses can come as:
        # - a list: [{...}]
        # - a dict with an 'outputs' list: {"outputs": [{...}]}
        # The actual predictions are often nested at: outputs[0]['predictions'] and may include
        # a nested 'predictions' dict plus 'predicted_classes'. We'll normalize to pred_obj.
        predictions_data = None
        predicted_classes = None
        class_confidences = {}

        # Normalize to a single workflow_item dict if possible
        workflow_item = None
        if isinstance(roboflow_result, dict) and 'outputs' in roboflow_result and isinstance(roboflow_result['outputs'], list) and len(roboflow_result['outputs']) > 0:
            workflow_item = roboflow_result['outputs'][0]
            app.logger.info(f"📋 Result contains 'outputs', using first output item")
            app.logger.info(f"📋 First output keys: {list(workflow_item.keys()) if isinstance(workflow_item, dict) else 'Not a dict'}")
        elif isinstance(roboflow_result, list) and len(roboflow_result) > 0:
            workflow_item = roboflow_result[0]
            app.logger.info(f"📋 Result is an array, using first item")
            app.logger.info(f"📋 First item keys: {list(workflow_item.keys()) if isinstance(workflow_item, dict) else 'Not a dict'}")
        elif isinstance(roboflow_result, dict):
            workflow_item = roboflow_result
            app.logger.info(f"📋 Result is a dict")
            app.logger.info(f"📋 Dict keys: {list(workflow_item.keys())}")
        else:
            app.logger.error(f"❌ Unexpected result type: {type(roboflow_result)}")
            sys.stdout.flush()
            return None

        # Try to locate the predictions object in known places
        pred_obj = None
        # Common: workflow_item has a 'predictions' key (which itself is a dict containing nested 'predictions')
        if isinstance(workflow_item, dict) and 'predictions' in workflow_item:
            pred_obj = workflow_item['predictions']
            app.logger.info(f"📋 Found 'predictions' key in workflow_item, type: {type(pred_obj)}")
        # Some responses embed predictions under workflow_item['outputs'] or similar; already normalized above.
        # If pred_obj is still None, check top-level keys for 'predictions'
        elif isinstance(workflow_item, dict):
            # As a fallback, search nested dicts for 'predictions' key
            for k, v in workflow_item.items():
                if isinstance(v, dict) and 'predictions' in v:
                    pred_obj = v
                    app.logger.info(f"📋 Found nested 'predictions' under key '{k}'")
                    break

        if pred_obj is None:
            app.logger.error(f"❌ Could not find a 'predictions' object in workflow result")
            sys.stdout.flush()
            return None

        # pred_obj is expected to be a dict that may contain:
        # - 'predicted_classes' (list)
        # - 'predictions' (dict) where keys are class names and values contain confidence
        if isinstance(pred_obj, dict):
            app.logger.info(f"📋 pred_obj keys: {list(pred_obj.keys())}")

            if 'predicted_classes' in pred_obj:
                predicted_classes = pred_obj['predicted_classes']
                app.logger.info(f"📋 Found 'predicted_classes': {predicted_classes}")

            # The actual class confidences are often in pred_obj['predictions']
            if 'predictions' in pred_obj and isinstance(pred_obj['predictions'], dict):
                predictions_dict = pred_obj['predictions']
                app.logger.info(f"📋 Found nested 'predictions' dict, keys: {list(predictions_dict.keys())}")
                # Extract edible and poisonous confidence scores
                for cls in ['edible', 'poisonous']:
                    if cls in predictions_dict:
                        val = predictions_dict[cls]
                        app.logger.info(f"📋 Found '{cls}' data: {val}")
                        if isinstance(val, dict) and 'confidence' in val:
                            class_confidences[cls] = float(val['confidence'])
                        elif isinstance(val, (int, float)):
                            class_confidences[cls] = float(val)
            # Fallback: if pred_obj itself is a list of predictions
            elif isinstance(pred_obj, list):
                predictions_data = pred_obj
                if len(predictions_data) > 0:
                    top_pred = predictions_data[0]
                    if isinstance(top_pred, dict):
                        if 'class' in top_pred:
                            predicted_classes = [top_pred['class']]
                        if 'confidence' in top_pred:
                            class_confidences[top_pred.get('class', 'unknown')] = float(top_pred['confidence'])
        sys.stdout.flush()
        
        # Determine the predicted class and confidence
        class_name = None
        confidence = None
        
        # Use predicted_classes if available (this is the most reliable)
        if predicted_classes and len(predicted_classes) > 0:
            class_name = predicted_classes[0]
            # Get confidence for the predicted class
            if class_name in class_confidences:
                confidence = class_confidences[class_name]
            else:
                # If we don't have confidence for predicted class, use the highest available
                if class_confidences:
                    confidence = max(class_confidences.values())
                    app.logger.warning(f"⚠️ Using highest confidence ({confidence}) since {class_name} confidence not found")
        else:
            # Fall back to highest confidence class if predicted_classes not available
            if class_confidences:
                class_name = max(class_confidences, key=class_confidences.get)
                confidence = class_confidences[class_name]
                app.logger.warning(f"⚠️ Using highest confidence class: {class_name} ({confidence})")
        
        if class_name is None or confidence is None:
            app.logger.error(f"❌ Could not extract class and confidence from result")
            sys.stdout.flush()
            return None
        
        app.logger.info(f"✅ Extracted classification: {class_name} (confidence: {confidence})")
        sys.stdout.flush()
        
        # Map Roboflow classes to our database
        roboflow_to_database = {
            'edible': {'edible': True, 'poisonous': False},
            'poisonous': {'edible': False, 'poisonous': True}
        }
        
        safety_info = roboflow_to_database.get(class_name.lower(), {'edible': False, 'poisonous': False})
        
        # Create result object
        result = {
            'id': 999,  # Special ID for Roboflow results
            'name': f"AI Classification: {class_name.title()}",
            'scientific_name': f"Roboflow Workflow Prediction",
            'edible': safety_info['edible'],
            'poisonous': safety_info['poisonous'],
            'taste': get_roboflow_taste_description(class_name),
            'habitat': 'AI Classification - Verify with experts',
            'season': 'AI Classification - Verify with experts',
            'confidence': round(confidence, 4),
            'uploaded_image': uploaded_image_data,
            'ml_class': class_name,
            'method': 'Roboflow Workflow AI',
            'all_confidences': class_confidences  # Include all confidence scores for debugging
        }
        
        return result
        
    except Exception as e:
        app.logger.error(f"❌ Error processing Roboflow result: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        sys.stdout.flush()
        return None

def get_roboflow_taste_description(class_name):
    """Get taste description for Roboflow classification"""
    if class_name == 'edible':
        return "✅ AI classified as edible - ALWAYS verify with experts before consumption!"
    elif class_name == 'poisonous':
        return "☠️ AI classified as poisonous - DO NOT CONSUME! Can cause severe illness or death!"
    else:
        return "❓ AI classification uncertain - Consult with mycologists before consumption"

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
    elif mushroom['edible']:
        safety_info["warning"] = "✅ Generally safe to eat when properly identified by experts"
    else:
        safety_info["warning"] = "❓ Unknown edibility - Always consult with mycologists before consumption"
    
    return jsonify(safety_info)


@app.route('/api/roboflow-identify', methods=['POST'])
def roboflow_identify():
    """Identify mushroom using only Roboflow model"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        
        # Perform Roboflow inference
        roboflow_result = roboflow_inference(image_file)
        if roboflow_result:
            processed_result = process_roboflow_result(roboflow_result, image)
            if processed_result:
                return jsonify({
                    "matches": [processed_result],
                    "message": "Roboflow AI identification complete - Always verify with experts before consumption!",
                    "method": "Roboflow AI"
                })
        
        return jsonify({"error": "Roboflow inference failed"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/kaggle-status', methods=['GET'])
def get_kaggle_status():
    """Get the status of the Roboflow API integration"""
    return jsonify({
        "dataset_available": True,
        "dataset_type": "Roboflow cloud-based classification",
        "total_samples": "Cloud-based dataset",
        "classes": ["edible", "poisonous"],
        "features": "Image-based classification",
        "safety_model_available": False,
        "image_model_available": False,
        "roboflow_available": True,
        "roboflow_workflow_id": ROBOFLOW_WORKFLOW_ID,
        "roboflow_workspace": ROBOFLOW_WORKSPACE,
        "roboflow_api_key_set": bool(ROBOFLOW_API_KEY),
        "workflow_endpoint": f"{ROBOFLOW_WORKFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}",
        "workflow_parameters": {
            "iou": ROBOFLOW_IOU,
            "max_detections": ROBOFLOW_MAX_DETECTIONS,
            "confidence": ROBOFLOW_CONFIDENCE
        },
        "message": "Using Roboflow Workflow API for mushroom identification"
    })

@app.route('/api/test-workflow', methods=['GET'])
def test_workflow():
    """Test endpoint to verify workflow configuration"""
    workflow_endpoint = f"{ROBOFLOW_WORKFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"
    return jsonify({
        "workspace": ROBOFLOW_WORKSPACE,
        "workflow_id": ROBOFLOW_WORKFLOW_ID,
        "workflow_endpoint": workflow_endpoint,
        "api_key_set": bool(ROBOFLOW_API_KEY),
        "api_key_preview": f"{ROBOFLOW_API_KEY[:10]}..." if ROBOFLOW_API_KEY else "Not set",
        "workflow_parameters": {
            "iOU": ROBOFLOW_IOU,
            "max_detections": ROBOFLOW_MAX_DETECTIONS,
            "confidence": ROBOFLOW_CONFIDENCE
        },
        "request_method": "POST",
        "request_format": "JSON",
        "request_body": {
            "api_key": "YOUR_API_KEY",
            "inputs": {
                "image": "base64_encoded_image_string",
                "iOU": ROBOFLOW_IOU,
                "max_detections": ROBOFLOW_MAX_DETECTIONS,
                "confidence": ROBOFLOW_CONFIDENCE
            }
        },
        "workflow_schema": {
            "inputs": [
                {"type": "WorkflowParameter", "name": "iOU", "default_value": 0.5},
                {"type": "WorkflowParameter", "name": "max_detections", "default_value": 50},
                {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
                {"type": "InferenceImage", "name": "image"}
            ]
        },
        "message": "Workflow endpoint configured. Test by uploading an image through the /api/identify endpoint."
    })


if __name__ == '__main__':
    init_sample_data()
    port = int(os.environ.get('PORT', 5001))
    # Enable debug mode to see detailed logs and auto-reload
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
