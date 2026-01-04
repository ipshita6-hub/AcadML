"""Model persistence utilities for saving and loading trained models"""

import joblib
import os
from typing import Any, Optional
from datetime import datetime
import json


class ModelPersistence:
    """Handle model saving and loading with metadata tracking"""
    
    def __init__(self, models_dir: str = 'models') -> None:
        """Initialize model persistence manager
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.metadata_file = os.path.join(models_dir, 'model_metadata.json')
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load existing metadata or create new"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self) -> None:
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(self, model: Any, model_name: str, accuracy: float, 
                   additional_info: Optional[dict] = None) -> str:
        """Save model with metadata
        
        Args:
            model: Trained model object
            model_name: Name of the model
            accuracy: Model accuracy score
            additional_info: Additional metadata to store
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.replace(' ', '_')}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        joblib.dump(model, filepath)
        
        # Store metadata
        self.metadata[model_name] = {
            'filepath': filepath,
            'accuracy': accuracy,
            'timestamp': timestamp,
            'additional_info': additional_info or {}
        }
        self._save_metadata()
        
        return filepath
    
    def load_model(self, model_name: str) -> Any:
        """Load model by name
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model object
        """
        if model_name not in self.metadata:
            raise ValueError(f"Model '{model_name}' not found in metadata")
        
        filepath = self.metadata[model_name]['filepath']
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        return joblib.load(filepath)
    
    def get_model_info(self, model_name: str) -> dict:
        """Get metadata for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata dictionary
        """
        if model_name not in self.metadata:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.metadata[model_name]
    
    def list_models(self) -> list:
        """List all saved models with their accuracies
        
        Returns:
            List of model information dictionaries
        """
        models_list = []
        for name, info in self.metadata.items():
            models_list.append({
                'name': name,
                'accuracy': info['accuracy'],
                'timestamp': info['timestamp']
            })
        return sorted(models_list, key=lambda x: x['accuracy'], reverse=True)
    
    def delete_model(self, model_name: str) -> None:
        """Delete a saved model
        
        Args:
            model_name: Name of the model to delete
        """
        if model_name not in self.metadata:
            raise ValueError(f"Model '{model_name}' not found")
        
        filepath = self.metadata[model_name]['filepath']
        if os.path.exists(filepath):
            os.remove(filepath)
        
        del self.metadata[model_name]
        self._save_metadata()
