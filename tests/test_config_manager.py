import unittest
import tempfile
import os
import yaml
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create a test config file
        self.test_config = {
            'data': {
                'sample_size': 500,
                'test_size': 0.25,
                'random_state': 123
            },
            'models': {
                'random_forest': {
                    'enabled': True,
                    'params': {'n_estimators': [100, 200]}
                },
                'svm': {
                    'enabled': False,
                    'params': {'C': [1, 10]}
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        os.rmdir(self.temp_dir)
    
    def test_load_config(self):
        """Test configuration loading"""
        config_manager = ConfigManager(self.config_file)
        
        # Check that config is loaded correctly
        self.assertEqual(config_manager.get('data.sample_size'), 500)
        self.assertEqual(config_manager.get('data.test_size'), 0.25)
        self.assertEqual(config_manager.get('data.random_state'), 123)
    
    def test_get_method(self):
        """Test get method with dot notation"""
        config_manager = ConfigManager(self.config_file)
        
        # Test existing keys
        self.assertEqual(config_manager.get('data.sample_size'), 500)
        self.assertTrue(config_manager.get('models.random_forest.enabled'))
        
        # Test non-existing keys with default
        self.assertEqual(config_manager.get('non.existing.key', 'default'), 'default')
        self.assertIsNone(config_manager.get('non.existing.key'))
    
    def test_set_method(self):
        """Test set method with dot notation"""
        config_manager = ConfigManager(self.config_file)
        
        # Set new value
        config_manager.set('data.sample_size', 1000)
        self.assertEqual(config_manager.get('data.sample_size'), 1000)
        
        # Set nested new value
        config_manager.set('new.nested.key', 'value')
        self.assertEqual(config_manager.get('new.nested.key'), 'value')
    
    def test_model_methods(self):
        """Test model-specific methods"""
        config_manager = ConfigManager(self.config_file)
        
        # Test model parameters
        rf_params = config_manager.get_model_params('Random Forest')
        self.assertEqual(rf_params, {'n_estimators': [100, 200]})
        
        # Test model enabled status
        self.assertTrue(config_manager.is_model_enabled('Random Forest'))
        self.assertFalse(config_manager.is_model_enabled('SVM'))
        
        # Test enabled models list
        enabled_models = config_manager.get_enabled_models()
        self.assertIn('Random Forest', enabled_models)
        self.assertNotIn('Svm', enabled_models)
    
    def test_validation(self):
        """Test configuration validation"""
        # Create invalid config
        invalid_config = {
            'data': {
                'sample_size': -100,  # Invalid: negative
                'test_size': 1.5      # Invalid: > 1
            },
            'cross_validation': {
                'cv_folds': 1         # Invalid: < 2
            }
        }
        
        invalid_file = os.path.join(self.temp_dir, 'invalid_config.yaml')
        with open(invalid_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(invalid_file)
        
        # Validation should fail
        self.assertFalse(config_manager.validate_config())
        
        # Clean up
        os.remove(invalid_file)
    
    def test_default_config(self):
        """Test default configuration when file doesn't exist"""
        non_existent_file = os.path.join(self.temp_dir, 'non_existent.yaml')
        config_manager = ConfigManager(non_existent_file)
        
        # Should load default config
        self.assertIsNotNone(config_manager.config)
        self.assertIn('data', config_manager.config)
        self.assertIn('sample_size', config_manager.config['data'])
    
    def test_save_config(self):
        """Test configuration saving"""
        config_manager = ConfigManager(self.config_file)
        
        # Modify config
        config_manager.set('data.sample_size', 2000)
        
        # Save to new file
        new_file = os.path.join(self.temp_dir, 'saved_config.yaml')
        config_manager.save_config(new_file)
        
        # Load and verify
        with open(new_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        self.assertEqual(saved_config['data']['sample_size'], 2000)
        
        # Clean up
        os.remove(new_file)

if __name__ == '__main__':
    unittest.main()