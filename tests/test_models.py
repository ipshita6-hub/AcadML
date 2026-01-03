import unittest
import numpy as np
import pandas as pd
import sys
import os
from sklearn.datasets import make_classification

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ModelTrainer
from data_loader import DataLoader

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.model_trainer = ModelTrainer()
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=5, n_classes=3, 
                                 n_informative=3, random_state=42)
        self.X_train = X[:80]
        self.X_test = X[80:]
        self.y_train = y[:80]
        self.y_test = y[80:]
        
        # Create target encoder mock
        self.target_encoder = type('MockEncoder', (), {
            'classes_': ['Class0', 'Class1', 'Class2'],
            'inverse_transform': lambda self, y: [f'Class{i}' for i in y]
        })()
    
    def test_model_initialization(self):
        """Test that all models are initialized correctly"""
        expected_models = ['Random Forest', 'Gradient Boosting', 'SVM', 
                          'Logistic Regression', 'KNN']
        
        self.assertEqual(list(self.model_trainer.models.keys()), expected_models)
        
        # Check that all models have required methods
        for name, model in self.model_trainer.models.items():
            self.assertTrue(hasattr(model, 'fit'))
            self.assertTrue(hasattr(model, 'predict'))
    
    def test_train_models(self):
        """Test model training"""
        self.model_trainer.train_models(self.X_train, self.y_train)
        
        # Check that trained models are stored
        self.assertEqual(len(self.model_trainer.trained_models), 5)
        
        # Check that all models are fitted
        for name, model in self.model_trainer.trained_models.items():
            # Should be able to make predictions
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.y_test))
    
    def test_evaluate_models(self):
        """Test model evaluation"""
        # First train the models
        self.model_trainer.train_models(self.X_train, self.y_train)
        
        # Then evaluate
        self.model_trainer.evaluate_models(self.X_test, self.y_test, self.target_encoder)
        
        # Check that results are stored
        self.assertEqual(len(self.model_trainer.results), 5)
        
        # Check that each result has required keys
        for name, result in self.model_trainer.results.items():
            self.assertIn('accuracy', result)
            self.assertIn('classification_report', result)
            self.assertIn('confusion_matrix', result)
            
            # Check accuracy is between 0 and 1
            self.assertGreaterEqual(result['accuracy'], 0)
            self.assertLessEqual(result['accuracy'], 1)
    
    def test_get_best_model(self):
        """Test best model selection"""
        # Train and evaluate models first
        self.model_trainer.train_models(self.X_train, self.y_train)
        self.model_trainer.evaluate_models(self.X_test, self.y_test, self.target_encoder)
        
        best_name, best_model = self.model_trainer.get_best_model()
        
        # Check that best model is returned
        self.assertIsInstance(best_name, str)
        self.assertIsNotNone(best_model)
        
        # Check that it's actually the best
        best_accuracy = self.model_trainer.results[best_name]['accuracy']
        for name, result in self.model_trainer.results.items():
            self.assertLessEqual(result['accuracy'], best_accuracy)
    
    def test_model_consistency(self):
        """Test that models produce consistent results"""
        # Train twice with same data
        self.model_trainer.train_models(self.X_train, self.y_train)
        predictions1 = {}
        for name, model in self.model_trainer.trained_models.items():
            predictions1[name] = model.predict(self.X_test)
        
        # Train again
        model_trainer2 = ModelTrainer()
        model_trainer2.train_models(self.X_train, self.y_train)
        predictions2 = {}
        for name, model in model_trainer2.trained_models.items():
            predictions2[name] = model.predict(self.X_test)
        
        # Random Forest and Gradient Boosting should be identical due to random_state
        np.testing.assert_array_equal(predictions1['Random Forest'], 
                                    predictions2['Random Forest'])
        np.testing.assert_array_equal(predictions1['Gradient Boosting'], 
                                    predictions2['Gradient Boosting'])

if __name__ == '__main__':
    unittest.main()