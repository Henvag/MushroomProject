# 🍄 Mushroom Project Prototype

A web application for identifying and learning about mushrooms, with a focus on safety and education.

## ✨ Features

- **Mushroom Database**: Browse 8 different mushroom species with detailed information
- **Safety Checking**: Verify if mushrooms are safe to consume
- **Identification Simulation**: Upload images for simulated mushroom identification
- **Filtering & Search**: Find mushrooms by properties (edible, poisonous)
- **Dark Mode**: Beautiful dark/light theme toggle
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Tech Stack

- **Frontend**: React.js with modern CSS
- **Backend**: Flask (Python) REST API
- **Data**: Kaggle Mushroom Classification Dataset (8,124 samples)
- **Styling**: CSS Variables for theming, responsive design
- **Frontend**: React.js with modern CSS
- **Backend**: Flask (Python) REST API
- **Data / ML**: Roboflow Workflow API (cloud-hosted image classification)
- **Styling**: CSS Variables for theming, responsive design

## 📊 Data & ML

The app uses a cloud-hosted image classification workflow (Roboflow Workflow API) for mushroom identification rather than a local Kaggle dataset.
- Data source: Roboflow-hosted dataset and model
- Classes: "edible" and "poisonous"
- Notes: The classification runs via the Roboflow Workflow API and returns class confidences. The backend also includes a safe simulation/fallback when the API is not available.

## 🏗️ Project Structure

```
INFO212Project/
├── app.py                 # Flask backend API
├── requirements.txt       # Python dependencies
├── download_kaggle_dataset.py  # Kaggle dataset downloader
├── public/               # Static assets
│   └── images/          # Mushroom images
├── src/                  # React frontend
│   ├── App.js           # Main React component
│   ├── App.css          # Component styles
│   └── index.js         # React entry point
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Python backend
source venv/bin/activate
pip install -r requirements.txt

# Node.js frontend
npm install
```

### 2. Configure Roboflow API key

The backend calls a Roboflow Workflow API. Instead of hardcoding secrets, set your Roboflow API key in the environment before starting the backend:

```bash
# macOS / Linux (zsh/bash)
export ROBOFLOW_API_KEY="your_roboflow_api_key_here"

# or use a .env loader in your shell if you prefer
```

### 3. Start the Application

```bash
# Terminal 1: Python backend
# 🍄 Mushroom Classifier

A concise, focused README explaining how the project works, how to configure it, and how to run it locally.

## What this project does

- Frontend: React app that lets a user upload a mushroom image.
- Backend: Flask API that accepts the image and forwards it to a Roboflow Workflow for classification.
- Output: the backend returns a single identification result (class: `edible` / `poisonous`) and a confidence score. The frontend displays the uploaded image, the predicted class, and the confidence percentage.

This app is for demonstration and educational purposes only — always consult a professional before making any safety or consumption decisions.

## Key details

- Image inference is performed by a Roboflow Workflow (cloud-hosted). The app posts the uploaded image to `/api/identify`, the backend calls the Roboflow Workflow API, and returns a normalized result.
- The backend includes a simulation fallback used only when Roboflow is unreachable. In normal development and production the Roboflow response is used.

## Configuration

1. Create and activate a Python virtualenv in the project root:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Roboflow API key in the environment:
```bash
export ROBOFLOW_API_KEY="<your_roboflow_api_key>"
```

Note: do NOT commit API keys to source control. Store them in environment variables or a secrets manager.

## Run the app locally

1. Start the Flask backend (default port 5001):
```bash
source venv/bin/activate
python app.py
```

2. In a separate terminal, start the React frontend:
```bash
npm install
npm start
```

Open the frontend at http://localhost:3000 — it proxies API requests to the backend at http://localhost:5001.

## API (minimal)

- POST /api/identify — multipart/form-data with key `image`. Returns JSON with `matches` (first match used by the UI), `method`, and `message`.
- GET /api/mushrooms — list of mushrooms used by the UI (sample data).
- POST /api/check-safety — checks safety for a mushroom id.
- GET /api/kaggle-status — (legacy) returns Roboflow/workflow status and configuration for diagnostic use.

The frontend expects the identify response to include at least:
- `matches[0].ml_class` (string, e.g., `poisonous`)
- `matches[0].confidence` (float 0..1) or `matches[0].all_confidences` mapping
- `matches[0].uploaded_image` (optional data URL for display)

## Testing

- Unit tests (Python): run `pytest` in the project root (ensure venv is active).

## Security & Privacy

- Images uploaded to the backend may be temporarily saved for inference. Do not upload private or sensitive images.
- Keep Roboflow API keys secret (use environment variables).

## Disclaimer

This project returns only a class label and confidence score. It does not provide a definitive identification. Always consult a qualified mycologist or medical professional for safety-critical decisions.

---

If you'd like the README shortened further or to include additional developer notes (tests, CI, hosting), tell me which sections to add or remove.
- Suggest new features
