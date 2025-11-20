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
cd /Users/henrik/INFO212Project
source venv/bin/activate
pip install -r requirements.txt
# ensure ROBOFLOW_API_KEY is set in your environment (see step 2)
python app.py

# Terminal 2: React frontend
cd /Users/henrik/INFO212Project
npm install
npm start
```

### 4. Open Your Browser
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5001

## 🔌 API Endpoints

- `GET /api/mushrooms` - Get all mushrooms
- `GET /api/mushrooms/<id>` - Get specific mushroom
-- `POST /api/identify` - Upload an image for identification (uses Roboflow Workflow API; falls back to simulation if unavailable)
- `POST /api/check-safety` - Check mushroom safety
- `GET /api/search?q=<query>` - Search mushrooms
-- `GET /api/kaggle-status` - (legacy) returns Roboflow/workflow status and configuration

## 🍄 Mushroom Species

1. **Chanterelle** - Edible, delicious
2. **Death Cap** - Extremely poisonous
3. **Fly Agaric** - Poisonous
4. **Psilocybe Cubensis** - Not recommended for consumption
5. **Porcini** - Edible, gourmet
6. **Destroying Angel** - Extremely poisonous
7. **Morel** - Edible, prized
8. **False Morel** - Poisonous

## 🎨 Features

### **Safety First**
- Clear warnings for poisonous mushrooms
- Detailed safety information

### **User Experience**
- Beautiful, modern interface
- Dark/light theme toggle
- Responsive design for all devices
- Intuitive navigation

### **Data Quality**
- Expert-verified mushroom information
- High-quality images
- Comprehensive safety warnings

## 🔒 Safety Disclaimer

**⚠️ IMPORTANT**: This application is for educational purposes only. 
- **Never consume mushrooms** based solely on app identification
- **Always consult experts** before eating wild mushrooms
- **When in doubt, throw it out**
- **Mushroom poisoning can be fatal**

## 🚧 Current Status

- ✅ **Basic mushroom database** - Complete
- ✅ **Safety checking** - Complete
- ✅ **Identification simulation** - Complete
- ✅ **Kaggle dataset integration** - Complete
- 🔄 **ML training** - Ready to implement
- 🔄 **Real image identification** - Future enhancement

## 🎯 Next Steps

1. **Train ML model** on Kaggle dataset for safety prediction
2. **Add more mushroom species** to the database
3. **Implement real image identification** with trained model
4. **Expand mushroom species coverage**

## 🤝 Contributing

This is a prototype project. Feel free to:
- Report bugs or issues
- Suggest new features
- Improve the UI/UX
- Add more mushroom data

## 📝 License

Educational project - use responsibly and always prioritize safety when dealing with mushrooms.

---

**🍄 Stay safe and happy mushroom hunting! (But don't eat anything you're not 100% sure about!)**
