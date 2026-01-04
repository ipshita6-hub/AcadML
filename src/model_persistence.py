import joblib
import pickle
import json
import os
from datetime import datetime
import numpy as np

class ModelPersistence:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model_with_metadata(self, model, model_name, metadata=None):
        """Save model with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'{model_name}_{timestamp}.pkl'
        model_path = os.path.join(self.model_dir, model_filename)
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        if metadata:
            metadata['saved_at'] = timestamp
            metadata['model_type'] = type(model).__name__
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            
            # Convert non-serializable objects
            metadata_serializable = self._make_serializable(metadata)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_serializable, f, indent=2)
        
        print(f"✅ Model saved: {model_path}")
        return model_path
    
    def load_model_with_metadata(self, model_path):
        """Load model with metadata"""
        model = joblib.load(model_path)
        
        # Load metadata if exists
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        metadata = None
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def list_saved_models(self):
        """List all saved models"""
        models = []
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.model_dir, filename)
                file_size = os.path.getsize(filepath)
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                models.append({
                    'filename': filename,
                    'path': filepath,
                    'size_mb': file_size / (1024 * 1024),
                    'saved_at': file_time.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return sorted(models, key=lambda x: x['saved_at'], reverse=True)
    
    def get_model_info(self, model_path):
        """Get information about a saved model"""
        model, metadata = self.load_model_with_metadata(model_path)
        
        info = {
            'model_type': type(model).__name__,
            'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'metadata': metadata
        }
        
        # Add model-specific info
        if hasattr(model, 'feature_importances_'):
            info['has_feature_importance'] = True
            info['n_features'] = len(model.feature_importances_)
        
        if hasattr(model, 'predict_proba'):
            info['has_probability'] = True
        
        if hasattr(model, 'n_estimators'):
            info['n_estimators'] = model.n_estimators
        
        return info
    
    def delete_old_models(self, keep_latest=3):
        """Delete old models, keeping only the latest N"""
        models = self.list_saved_models()
        
        if len(models) <= keep_latest:
            print(f"Only {len(models)} models found. Keeping all.")
            return
        
        models_to_delete = models[keep_latest:]
        
        for model_info in models_to_delete:
            os.remove(model_info['path'])
            metadata_path = model_info['path'].replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            print(f"🗑️  Deleted: {model_info['filename']}")
    
    def export_model_summary(self, output_path='results/model_summary.txt'):
        """Export summary of all saved models"""
        models = self.list_saved_models()
        
        with open(output_path, 'w') as f:
            f.write("SAVED MODELS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            for i, model_info in enumerate(models, 1):
                f.write(f"{i}. {model_info['filename']}\n")
                f.write(f"   Size: {model_info['size_mb']:.2f} MB\n")
                f.write(f"   Saved: {model_info['saved_at']}\n")
                
                # Get additional info
                try:
                    info = self.get_model_info(model_info['path'])
                    f.write(f"   Type: {info['model_type']}\n")
                    if 'n_features' in info:
                        f.write(f"   Features: {info['n_features']}\n")
                    if 'n_estimators' in info:
                        f.write(f"   Estimators: {info['n_estimators']}\n")
                except:
                    pass
                
                f.write("\n")
        
        print(f"✅ Model summary exported to {output_path}")
        return output_path
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj