import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

class ModelPredictor:
    def __init__(self, model_path, scaler=None, label_encoders=None, target_encoder=None, feature_names=None):
        """Initialize predictor with trained model and preprocessing objects"""
        self.model = joblib.load(model_path)
        self.scaler = scaler
        self.label_encoders = label_encoders or {}
        self.target_encoder = target_encoder
        self.feature_names = feature_names
        
    def predict_single(self, input_dict):
        """Make prediction for a single sample"""
        # Convert dict to DataFrame
        df = pd.DataFrame([input_dict])
        
        # Preprocess
        X = self._preprocess_input(df)
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            probability = max(proba)
        
        # Decode prediction
        if self.target_encoder:
            prediction_label = self.target_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = prediction
        
        return {
            'prediction': prediction_label,
            'confidence': probability,
            'raw_prediction': prediction
        }
    
    def predict_batch(self, input_df):
        """Make predictions for multiple samples"""
        X = self._preprocess_input(input_df)
        predictions = self.model.predict(X)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            proba_matrix = self.model.predict_proba(X)
            probabilities = np.max(proba_matrix, axis=1)
        
        # Decode predictions
        if self.target_encoder:
            prediction_labels = self.target_encoder.inverse_transform(predictions)
        else:
            prediction_labels = predictions
        
        results_df = pd.DataFrame({
            'prediction': prediction_labels,
            'confidence': probabilities if probabilities is not None else [None] * len(predictions),
            'raw_prediction': predictions
        })
        
        return results_df
    
    def _preprocess_input(self, df):
        """Preprocess input data"""
        df_processed = df.copy()
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df_processed.columns:
                df_processed[col] = encoder.transform(df_processed[col])
        
        # Select features in correct order
        if self.feature_names:
            X = df_processed[self.feature_names].values
        else:
            X = df_processed.values
        
        # Scale features
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X
    
    def get_feature_importance_explanation(self, feature_name):
        """Get explanation for a specific feature's importance"""
        if not hasattr(self.model, 'feature_importances_'):
            return f"Feature importance not available for this model type"
        
        if self.feature_names and feature_name in self.feature_names:
            idx = self.feature_names.index(feature_name)
            importance = self.model.feature_importances_[idx]
            
            # Rank among all features
            all_importances = self.model.feature_importances_
            rank = np.argsort(all_importances)[::-1].tolist().index(idx) + 1
            
            return {
                'feature': feature_name,
                'importance_score': importance,
                'rank': rank,
                'total_features': len(self.feature_names),
                'percentile': (rank / len(self.feature_names)) * 100
            }
        
        return f"Feature '{feature_name}' not found"
    
    def explain_prediction(self, input_dict):
        """Provide detailed explanation for a prediction"""
        prediction_result = self.predict_single(input_dict)
        
        explanation = {
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'input_features': input_dict,
            'feature_importance': {}
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            for feature in self.feature_names:
                explanation['feature_importance'][feature] = self.get_feature_importance_explanation(feature)
        
        return explanation
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names) if self.feature_names else 'Unknown',
            'features': self.feature_names,
            'has_feature_importance': hasattr(self.model, 'feature_importances_'),
            'has_probability': hasattr(self.model, 'predict_proba'),
            'target_classes': self.target_encoder.classes_.tolist() if self.target_encoder else 'Unknown'
        }