import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Camera, Database, Shield, Search, Filter, Sun, Moon } from 'lucide-react';
import './App.css';

function App() {
  console.log('App component rendering...'); // Debug log
  
  const [activeTab, setActiveTab] = useState('identify');
  const [mushrooms, setMushrooms] = useState([]);
  const [filteredMushrooms, setFilteredMushrooms] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');
  const [selectedFile, setSelectedFile] = useState(null);
  const [identificationResults, setIdentificationResults] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [kaggleStatus, setKaggleStatus] = useState(null);

  console.log('State initialized:', { mushrooms, isDarkMode }); // Debug log

  useEffect(() => {
    // Check system preference and initialize theme
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('theme');
    
    if (savedTheme) {
      setIsDarkMode(savedTheme === 'dark');
    } else {
      setIsDarkMode(prefersDark);
    }
  }, []);

  useEffect(() => {
    // Apply theme to document
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  useEffect(() => {
    // Check if we're in production (GitHub Pages) or development
    const isProduction = window.location.hostname === 'mushroomproject.app';
    const apiBaseUrl = isProduction ? 'https://your-backend-url.railway.app' : '';
    
    // Set up axios base URL for production
    if (isProduction) {
      axios.defaults.baseURL = apiBaseUrl;
    }
    
    // Load sample data for demo purposes
    const sampleMushrooms = [
      {
        id: 1,
        name: "Chanterelle",
        scientific_name: "Cantharellus cibarius",
        edible: true,
        poisonous: false,
        psychedelic: false,
        taste: "Delicate, slightly peppery with fruity apricot aroma",
        habitat: "Forest floors, often near oak and pine trees",
        season: "Summer to Fall",
        confidence: 0.85,
        image_url: "/images/chanterelle.jpg"
      },
      {
        id: 2,
        name: "Death Cap",
        scientific_name: "Amanita phalloides",
        edible: false,
        poisonous: true,
        psychedelic: false,
        taste: "DO NOT EAT - Extremely toxic",
        habitat: "Under oak and beech trees",
        season: "Summer to Fall",
        confidence: 0.90,
        image_url: "/images/death-cap.jpg"
      },
      {
        id: 3,
        name: "Fly Agaric",
        scientific_name: "Amanita muscaria",
        edible: false,
        poisonous: true,
        psychedelic: true,
        taste: "DO NOT EAT - Contains muscimol and ibotenic acid",
        habitat: "Under birch, pine, and spruce trees",
        season: "Summer to Fall",
        confidence: 0.92,
        image_url: "/images/fly-agaric.jpg"
      },
      {
        id: 4,
        name: "Porcini",
        scientific_name: "Boletus edulis",
        edible: true,
        poisonous: false,
        psychedelic: false,
        taste: "Rich, nutty, and meaty flavor",
        habitat: "Coniferous and deciduous forests",
        season: "Summer to Fall",
        confidence: 0.88,
        image_url: "/images/porcini.jpg"
      },
      {
        id: 5,
        name: "Morel",
        scientific_name: "Morchella esculenta",
        edible: true,
        poisonous: false,
        psychedelic: false,
        taste: "Earthy, nutty, and meaty flavor",
        habitat: "Forests, especially after fires",
        season: "Spring",
        confidence: 0.82,
        image_url: "/images/morel.jpg"
      }
    ];
    
    setMushrooms(sampleMushrooms);
    
    // Try to fetch from API if backend is available
    if (!isProduction) {
      fetchMushrooms();
      checkKaggleStatus();
    }
  }, []);

  useEffect(() => {
    filterMushrooms();
  }, [mushrooms, searchQuery, activeFilter]);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  const filterMushrooms = useCallback(() => {
    let filtered = mushrooms;

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter(mushroom =>
        mushroom.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        mushroom.scientific_name.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply property filter
    if (activeFilter !== 'all') {
      filtered = filtered.filter(mushroom => mushroom[activeFilter]);
    }

    setFilteredMushrooms(filtered);
  }, [mushrooms, searchQuery, activeFilter]);

  const checkKaggleStatus = async () => {
    try {
      const response = await axios.get('/api/kaggle-status');
      setKaggleStatus(response.data);
    } catch (err) {
      console.log('Kaggle status check failed:', err);
      setKaggleStatus({ dataset_available: false });
    }
  };


  const fetchMushrooms = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/mushrooms');
      setMushrooms(response.data);
    } catch (err) {
      setError('Failed to load mushrooms');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Simple fallback render for debugging
  if (mushrooms.length === 0) {
    console.log('Rendering fallback...'); // Debug log
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h1>Loading Mushroom App...</h1>
        <p>If you see this, the component is mounting but data is loading</p>
      </div>
    );
  }

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setIdentificationResults(null); // Clear previous results
    }
  };

  const identifyMushroom = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    try {
      setLoading(true);
      setError('');
      setIdentificationResults(null);
      
      // Check if we're in production (no backend available)
      const isProduction = window.location.hostname === 'mushroomproject.app';
      
      if (isProduction) {
        // Simulate identification for demo purposes
        await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time
        
        const simulatedResults = {
          matches: [
            {
              id: 1,
              name: "Chanterelle",
              scientific_name: "Cantharellus cibarius",
              edible: true,
              poisonous: false,
              psychedelic: false,
              taste: "Delicate, slightly peppery with fruity apricot aroma",
              habitat: "Forest floors, often near oak and pine trees",
              season: "Summer to Fall",
              confidence: 0.85,
              uploaded_image: URL.createObjectURL(selectedFile),
              image_url: "/images/chanterelle.jpg"
            }
          ],
          message: "Demo identification - Upload your image to see AI-powered identification!",
          method: "Demo Mode"
        };
        
        setIdentificationResults(simulatedResults);
        return;
      }
      
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('image', selectedFile);
      
      // Send image to backend for ML identification
      const response = await axios.post('/api/identify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setIdentificationResults(response.data);
    } catch (err) {
      setError('Failed to identify mushroom');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const checkSafety = async (mushroomId) => {
    try {
      const response = await axios.post('/api/check-safety', {
        mushroom_id: mushroomId
      });
      
      alert(`Safety Check: ${response.data.warning}`);
    } catch (err) {
      setError('Failed to check safety');
    }
  };



  const renderIdentificationTab = () => (
    <div className="card">
      <div className="card-header">
        <Camera size={28} className="card-icon" />
        <div>
          <h2>Identify Mushroom</h2>
          <p>Upload a photo to identify mushrooms (simulation mode)</p>
        </div>
      </div>

      {/* Model Status */}
      {kaggleStatus && (
        <div className={`kaggle-status ${kaggleStatus.safety_model_available || kaggleStatus.image_model_available || kaggleStatus.roboflow_available ? 'kaggle-available' : 'kaggle-unavailable'}`}>
          <Database size={20} />
          <span>
            Models: {kaggleStatus.message}
            {kaggleStatus.safety_model_available && ' • Safety Model Active'}
            {kaggleStatus.image_model_available && ' • Local Image Classification Active'}
            {kaggleStatus.roboflow_available && ' • Roboflow AI Active'}
          </span>
        </div>
      )}
      
      <div className="file-upload" onClick={() => document.getElementById('file-input').click()}>
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
        />
        {selectedFile ? (
          <div className="file-preview">
            <p className="file-name">Selected: {selectedFile.name}</p>
            <img 
              src={URL.createObjectURL(selectedFile)} 
              alt="Preview" 
              className="file-preview-image"
            />
          </div>
        ) : (
          <div className="upload-placeholder">
            <Camera size={64} className="upload-icon" />
            <h3>Upload Mushroom Photo</h3>
            <p>Click to select an image file</p>
            <span className="upload-hint">Supports JPG, PNG up to 10MB</span>
          </div>
        )}
      </div>

      <button 
        className="btn btn-primary" 
        onClick={identifyMushroom}
        disabled={!selectedFile || loading}
      >
        {loading ? 'Identifying...' : 'Identify Mushroom'}
      </button>

      {identificationResults && (
        <div className="results-section">
          <div className="results-header">
            <h3>Identification Results</h3>
            <div className={`method-badge ${
              identificationResults.method === 'Roboflow AI' ? 'method-roboflow' :
              identificationResults.method === 'Local ML Model' ? 'method-ml' : 
              'method-sim'
            }`}>
              {identificationResults.method === 'Roboflow AI' ? <Database size={16} /> :
               identificationResults.method === 'Local ML Model' ? <Database size={16} /> : 
               <Database size={16} />}
              {identificationResults.method}
            </div>
          </div>
          
          <div className="mushroom-grid">
            {identificationResults.matches.map((mushroom, index) => (
              <div key={index} className="mushroom-card">
                <img 
                  src={mushroom.uploaded_image || mushroom.image_url} 
                  alt={mushroom.name}
                  className="mushroom-image"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = 'https://via.placeholder.com/400x300/cccccc/666666?text=' + encodeURIComponent(mushroom.name);
                  }}
                />
                <div className="mushroom-info">
                  <div className="mushroom-header">
                    <div className="mushroom-name">{mushroom.name}</div>
                    <div className="mushroom-scientific">{mushroom.scientific_name}</div>
                    {mushroom.ml_class && (
                      <div className="ml-class">ML Class: {mushroom.ml_class.replace(/_/g, ' ')}</div>
                    )}
                  </div>
                  
                  <div className="mushroom-properties">
                    {mushroom.edible && <span className="property-badge property-edible">Edible</span>}
                    {mushroom.poisonous && <span className="property-badge property-poisonous">Poisonous</span>}
                    {mushroom.psychedelic && <span className="property-badge property-psychedelic">Psychedelic</span>}
                  </div>

                  <div className="confidence-section">
                    <div className="confidence-label">
                      <strong>Confidence:</strong> {Math.round(mushroom.confidence * 100)}%
                    </div>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill" 
                        style={{ width: `${mushroom.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="mushroom-details">
                    <div className="detail-item">
                      <strong>Taste:</strong> {mushroom.taste}
                    </div>
                  </div>
                  
                  <button 
                    className="btn btn-secondary" 
                    onClick={() => checkSafety(mushroom.id)}
                  >
                    <Shield size={16} /> Check Safety
                  </button>
                </div>
              </div>
            ))}
          </div>
          
          {identificationResults.message && (
            <div className="identification-message">
              {identificationResults.message}
            </div>
          )}
        </div>
      )}
    </div>
  );

  const renderBrowseTab = () => (
    <div className="card">
      <div className="card-header">
        <Database size={28} className="card-icon" />
        <div>
          <h2>Browse Mushrooms</h2>
          <p>Search and filter mushrooms by their properties</p>
        </div>
      </div>

      <div className="search-container">
        <div className="search-box">
          <Search size={20} className="search-icon" />
          <input
            type="text"
            className="form-control search-input"
            placeholder="Search mushrooms by name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="filters">
          <div className="filter-label">
            <Filter size={16} />
            <span>Filter by:</span>
          </div>
          <button 
            className={`filter-btn ${activeFilter === 'all' ? 'active' : ''}`}
            onClick={() => setActiveFilter('all')}
          >
            All
          </button>
          <button 
            className={`filter-btn ${activeFilter === 'edible' ? 'active' : ''}`}
            onClick={() => setActiveFilter('edible')}
          >
            Edible
          </button>
          <button 
            className={`filter-btn ${activeFilter === 'poisonous' ? 'active' : ''}`}
            onClick={() => setActiveFilter('poisonous')}
          >
            Poisonous
          </button>
          <button 
            className={`filter-btn ${activeFilter === 'psychedelic' ? 'active' : ''}`}
            onClick={() => setActiveFilter('psychedelic')}
          >
            Psychedelic
          </button>
        </div>
      </div>

      {loading ? (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Loading mushrooms...</p>
        </div>
      ) : (
        <div className="mushroom-grid">
          {filteredMushrooms.map(mushroom => (
            <div key={mushroom.id} className="mushroom-card">
              <img 
                src={mushroom.image_url} 
                alt={mushroom.name}
                className="mushroom-image"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = 'https://via.placeholder.com/400x300/cccccc/666666?text=' + encodeURIComponent(mushroom.name);
                }}
              />
              <div className="mushroom-info">
                <div className="mushroom-header">
                  <div className="mushroom-name">{mushroom.name}</div>
                  <div className="mushroom-scientific">{mushroom.scientific_name}</div>
                </div>
                
                <div className="mushroom-properties">
                  {mushroom.edible && <span className="property-badge property-edible">Edible</span>}
                  {mushroom.poisonous && <span className="property-badge property-poisonous">Poisonous</span>}
                  {mushroom.psychedelic && <span className="property-badge property-psychedelic">Psychedelic</span>}
                </div>

                <div className="mushroom-details">
                  <div className="detail-item">
                    <strong>Habitat:</strong> {mushroom.habitat}
                  </div>
                  <div className="detail-item">
                    <strong>Season:</strong> {mushroom.season}
                  </div>
                  <div className="detail-item">
                    <strong>Taste:</strong> {mushroom.taste}
                  </div>
                </div>

                <button 
                  className="btn btn-secondary" 
                  onClick={() => checkSafety(mushroom.id)}
                >
                  <Shield size={16} /> Check Safety
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {filteredMushrooms.length === 0 && !loading && (
        <div className="empty-state">
          <Database size={64} className="empty-icon" />
          <h3>No mushrooms found</h3>
          <p>Try adjusting your search or filters</p>
        </div>
      )}
    </div>
  );

  return (
    <div className="App">
      <div className="container">
        <div className="header">
          <div className="header-content">
            <h1>🍄 Mushroom Project Prototype</h1>
            <p>Identify edible, poisonous, and psychedelic mushrooms safely</p>
          </div>
          <div className="theme-toggle">
            <button 
              className="theme-toggle-btn"
              onClick={toggleDarkMode}
              aria-label={`Switch to ${isDarkMode ? 'light' : 'dark'} mode`}
            >
              {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>
        </div>

        <div className="navigation">
          <button 
            className={`nav-link ${activeTab === 'identify' ? 'active' : ''}`}
            onClick={() => setActiveTab('identify')}
          >
            <Camera size={20} /> Identify
          </button>
          <button 
            className={`nav-link ${activeTab === 'browse' ? 'active' : ''}`}
            onClick={() => setActiveTab('browse')}
          >
            <Database size={20} /> Browse
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {activeTab === 'identify' && renderIdentificationTab()}
        {activeTab === 'browse' && renderBrowseTab()}
      </div>
    </div>
  );
}

export default App;
