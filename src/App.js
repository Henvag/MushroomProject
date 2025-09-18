import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Camera, Database, Shield, Search, Filter, Sun, Moon, Brain } from 'lucide-react';
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
  const [featureCodes, setFeatureCodes] = useState(null);
  const [mlPrediction, setMlPrediction] = useState(null);
  const [selectedFeatures, setSelectedFeatures] = useState({});

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
    // Temporarily comment out API calls to test rendering
    // fetchMushrooms();
    // checkKaggleStatus();
    // fetchFeatureCodes();
    
    // Set some dummy data for testing
    setMushrooms([
      {
        id: 1,
        name: "Test Mushroom",
        scientific_name: "Testus mushroomus",
        edible: true,
        poisonous: false,
        psychedelic: false,
        image_url: "/images/chanterelle.jpg"
      }
    ]);
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

  const fetchFeatureCodes = async () => {
    try {
      const response = await axios.get('/api/feature-codes');
      setFeatureCodes(response.data);
    } catch (err) {
      console.log('Feature codes fetch failed:', err);
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

  const handleFeatureChange = (feature, value) => {
    setSelectedFeatures(prev => ({
      ...prev,
      [feature]: value
    }));
  };

  const predictSafety = async () => {
    if (Object.keys(selectedFeatures).length < 22) {
      setError('Please select all 22 features for accurate prediction');
      return;
    }

    try {
      setLoading(true);
      setError('');
      setMlPrediction(null);

      const response = await axios.post('/api/predict-safety', {
        features: selectedFeatures
      });

      setMlPrediction(response.data);
    } catch (err) {
      if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else {
        setError('Failed to predict safety');
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const renderMLSafetyTab = () => (
    <div className="card">
      <div className="card-header">
        <Brain size={28} className="card-icon" />
        <div>
          <h2>ML Safety Prediction</h2>
          <p>Use the trained model to predict mushroom safety from features</p>
        </div>
      </div>

      {/* ML Status */}
      {kaggleStatus && (
        <div className={`ml-status ${kaggleStatus.safety_model_available ? 'ml-available' : 'ml-unavailable'}`}>
          <Brain size={20} />
          <span>
            {kaggleStatus.safety_model_available 
              ? `Safety Prediction Model Active - ${kaggleStatus.total_samples} samples, ${kaggleStatus.features} features`
              : 'Safety Prediction Model not available - Please train the model first'
            }
          </span>
        </div>
      )}

      {featureCodes && (
        <div className="feature-selection">
          <h3>Select Mushroom Features</h3>
          <p>Choose the characteristics of the mushroom you want to analyze:</p>
          
          <div className="feature-grid">
            {Object.entries(featureCodes).map(([feature, codes]) => (
              <div key={feature} className="feature-item">
                <label className="feature-label">
                  {feature.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                </label>
                <select
                  className="feature-select"
                  value={selectedFeatures[feature] || ''}
                  onChange={(e) => handleFeatureChange(feature, e.target.value)}
                >
                  <option value="">Select...</option>
                  {Object.entries(codes).map(([code, description]) => (
                    <option key={code} value={code}>
                      {code} - {description}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          <div className="feature-actions">
            <button 
              className="btn btn-primary" 
              onClick={predictSafety}
              disabled={Object.keys(selectedFeatures).length < 22 || loading}
            >
              {loading ? 'Predicting...' : 'Predict Safety'}
            </button>
            
            <button 
              className="btn btn-secondary" 
              onClick={() => setSelectedFeatures({})}
            >
              Clear All
            </button>
          </div>

          <div className="feature-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${(Object.keys(selectedFeatures).length / 22) * 100}%` }}
              ></div>
            </div>
            <span className="progress-text">
              {Object.keys(selectedFeatures).length} / 22 features selected
            </span>
          </div>
        </div>
      )}

      {/* ML Prediction Results */}
      {mlPrediction && (
        <div className="ml-results">
          <div className="results-header">
            <h3>ML Safety Prediction Results</h3>
            <div className={`safety-badge ${mlPrediction.safety === 'edible' ? 'safety-edible' : 'safety-poisonous'}`}>
              {mlPrediction.safety === 'edible' ? '✅ EDIBLE' : '☠️ POISONOUS'}
            </div>
          </div>

          <div className="prediction-details">
            <div className="confidence-section">
              <div className="confidence-label">
                <strong>Confidence:</strong> {Math.round(mlPrediction.confidence * 100)}%
              </div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{ width: `${mlPrediction.confidence * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="probabilities">
              <h4>Class Probabilities:</h4>
              <div className="probability-grid">
                {Object.entries(mlPrediction.probabilities).map(([class_name, prob]) => (
                  <div key={class_name} className="probability-item">
                    <span className="class-name">
                      {class_name === 'e' ? 'Edible' : 'Poisonous'}:
                    </span>
                    <span className="probability-value">
                      {Math.round(prob * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="safety-warning">
              <strong>⚠️ Safety Warning:</strong>
              <p>{mlPrediction.warning}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );

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
              {identificationResults.method === 'Roboflow AI' ? <Brain size={16} /> :
               identificationResults.method === 'Local ML Model' ? <Brain size={16} /> : 
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
            className={`nav-link ${activeTab === 'ml-safety' ? 'active' : ''}`}
            onClick={() => setActiveTab('ml-safety')}
          >
            <Brain size={20} /> ML Safety
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
        {activeTab === 'ml-safety' && renderMLSafetyTab()}
        {activeTab === 'browse' && renderBrowseTab()}
      </div>
    </div>
  );
}

export default App;
