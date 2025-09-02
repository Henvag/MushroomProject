# 🍄 Mushroom Project Prototype

A web application for identifying and learning about mushrooms, with a focus on safety and education.

## ✨ Features

- **Mushroom Database**: Browse 8 different mushroom species with detailed information
- **Safety Checking**: Verify if mushrooms are safe to consume
- **Identification Simulation**: Upload images for simulated mushroom identification
- **Filtering & Search**: Find mushrooms by properties (edible, poisonous, psychedelic)
- **Dark Mode**: Beautiful dark/light theme toggle
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Tech Stack

- **Frontend**: React.js with modern CSS
- **Backend**: Flask (Python) REST API
- **Data**: Kaggle Mushroom Classification Dataset (8,124 samples)
- **Styling**: CSS Variables for theming, responsive design

## 📊 Dataset

The app uses the **Kaggle Mushroom Classification Dataset**:
- **Total samples**: 8,124 mushrooms
- **Classes**: Edible vs Poisonous
- **Features**: 23 different characteristics (cap shape, gill color, odor, etc.)
- **Safety focus**: High accuracy for identifying dangerous mushrooms

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

### 2. Download Kaggle Dataset
```bash
python download_kaggle_dataset.py
```

### 3. Start the Application
```bash
# Terminal 1: Start Flask backend
source venv/bin/activate
python app.py

# Terminal 2: Start React frontend
npm start
```

### 4. Open Your Browser
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5001

## 🔌 API Endpoints

- `GET /api/mushrooms` - Get all mushrooms
- `GET /api/mushrooms/<id>` - Get specific mushroom
- `POST /api/identify` - Simulate mushroom identification
- `POST /api/check-safety` - Check mushroom safety
- `GET /api/search?q=<query>` - Search mushrooms
- `GET /api/kaggle-status` - Get dataset status

## 🍄 Mushroom Species

1. **Chanterelle** - Edible, delicious
2. **Death Cap** - Extremely poisonous
3. **Fly Agaric** - Poisonous, psychedelic
4. **Psilocybe Cubensis** - Psychedelic
5. **Porcini** - Edible, gourmet
6. **Destroying Angel** - Extremely poisonous
7. **Morel** - Edible, prized
8. **False Morel** - Poisonous

## 🎨 Features

### **Safety First**
- Clear warnings for poisonous mushrooms
- Detailed safety information
- Psychedelic mushroom identification

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
4. **Expand psychedelic mushroom coverage**

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
