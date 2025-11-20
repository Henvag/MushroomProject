import pytest
import json
import tempfile
import os
from app import app, load_mushroom_data, save_mushroom_data, get_safety_info_for_class

class TestEssentialFunctions:
    """Essential tests for the mushroom identification app"""
    
    def test_mushroom_data_management(self):
        """Test loading and saving mushroom data - core functionality"""
        # Sample test data
        test_data = [
            {
                "id": 1,
                "name": "Test Mushroom",
                "scientific_name": "Testus mushroomus",
                "edible": True,
                "poisonous": False,
                "taste": "Test taste",
                "habitat": "Test habitat",
                "season": "Test season",
                "confidence": 0.85,
                "image_url": "/images/test.jpg"
            }
        ]
        
        # Backup original file if it exists
        original_file = 'mushroom_data.json'
        backup_file = original_file + '.backup'
        
        if os.path.exists(original_file):
            os.rename(original_file, backup_file)
        
        try:
            # Test saving data
            save_mushroom_data(test_data)
            assert os.path.exists(original_file)
            
            # Test loading data
            loaded_data = load_mushroom_data()
            assert len(loaded_data) == 1
            assert loaded_data[0]['name'] == 'Test Mushroom'
            assert loaded_data[0]['edible'] == True
            assert loaded_data[0]['poisonous'] == False
            
        finally:
            # Cleanup
            if os.path.exists(original_file):
                os.remove(original_file)
            if os.path.exists(backup_file):
                os.rename(backup_file, original_file)
    
    def test_safety_classification_system(self):
        """Test the safety classification system - critical for user safety"""
        # Test edible mushroom classification
        chanterelle_safety = get_safety_info_for_class('chanterelle')
        assert chanterelle_safety['edible'] == True
        assert chanterelle_safety['poisonous'] == False
        assert 'safe' in chanterelle_safety['warning'].lower()
        
        # Test poisonous mushroom classification
        death_cap_safety = get_safety_info_for_class('death_cap')
        assert death_cap_safety['edible'] == False
        assert death_cap_safety['poisonous'] == True
        assert 'deadly' in death_cap_safety['warning'].lower()
        
        
        # Test unknown mushroom classification
        unknown_safety = get_safety_info_for_class('unknown_mushroom')
        assert unknown_safety['edible'] == False
        assert unknown_safety['poisonous'] == False
        assert 'unknown' in unknown_safety['warning'].lower()
