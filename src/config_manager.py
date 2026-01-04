import yaml
import os
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"✅ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️  Configuration file {self.config_path} not found. Using default settings.")
            return self.get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing configuration file: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            'data': {
                'sample_size': 1000,
                'test_size': 0.2,
                'random_state': 42,
                'target_column': 'performance'
            },
            'cross_validation': {
                'cv_folds': 5,
                'random_state': 42
            },
            'hyperparameter_tuning': {
                'method': 'randomized',
                'cv_folds': 3,
                'n_iter': 20,
                'n_jobs': -1,
                'random_state': 42
            },
            'visualization': {
                'save_plots': True,
                'plot_format': 'png',
                'dpi': 300,
                'interactive_plots': True,
                'results_dir': 'results'
            },
            'evaluation': {
                'metrics': ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
                'ensemble_weights': {
                    'accuracy': 0.4,
                    'f1_macro': 0.3,
                    'balanced_accuracy': 0.3
                }
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_file': 'logs/academic_ml.log'
            },
            'output': {
                'save_models': True,
                'model_dir': 'models',
                'save_results': True,
                'generate_report': True
            }
        }
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'data.sample_size')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the value
        config_section[keys[-1]] = value
    
    def save_config(self, path=None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            print(f"✅ Configuration saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def create_directories(self):
        """Create necessary directories based on configuration"""
        directories = [
            self.get('visualization.results_dir', 'results'),
            self.get('output.model_dir', 'models'),
            os.path.dirname(self.get('logging.log_file', 'logs/academic_ml.log'))
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
                print(f"📁 Created directory: {directory}")
    
    def validate_config(self):
        """Validate configuration values"""
        validation_errors = []
        
        # Validate data configuration
        sample_size = self.get('data.sample_size')
        if not isinstance(sample_size, int) or sample_size <= 0:
            validation_errors.append("data.sample_size must be a positive integer")
        
        test_size = self.get('data.test_size')
        if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
            validation_errors.append("data.test_size must be between 0 and 1")
        
        # Validate cross-validation configuration
        cv_folds = self.get('cross_validation.cv_folds')
        if not isinstance(cv_folds, int) or cv_folds < 2:
            validation_errors.append("cross_validation.cv_folds must be an integer >= 2")
        
        # Validate hyperparameter tuning
        hp_method = self.get('hyperparameter_tuning.method')
        if hp_method not in ['grid', 'randomized']:
            validation_errors.append("hyperparameter_tuning.method must be 'grid' or 'randomized'")
        
        # Validate visualization
        plot_format = self.get('visualization.plot_format')
        if plot_format not in ['png', 'jpg', 'svg', 'pdf']:
            validation_errors.append("visualization.plot_format must be one of: png, jpg, svg, pdf")
        
        if validation_errors:
            print("❌ Configuration validation errors:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        else:
            print("✅ Configuration validation passed")
            return True
    
    def print_config(self):
        """Print current configuration in a readable format"""
        print("\n⚙️  Current Configuration:")
        print("=" * 50)
        self._print_dict(self.config, indent=0)
    
    def _print_dict(self, d, indent=0):
        """Recursively print dictionary with indentation"""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    def get_model_params(self, model_name):
        """Get parameter grid for a specific model"""
        model_key = model_name.lower().replace(' ', '_')
        return self.get(f'models.{model_key}.params', {})
    
    def is_model_enabled(self, model_name):
        """Check if a model is enabled in configuration"""
        model_key = model_name.lower().replace(' ', '_')
        return self.get(f'models.{model_key}.enabled', True)
    
    def get_enabled_models_summary(self):
        """Get summary of enabled models with their configurations"""
        models_config = self.get('models', {})
        enabled_count = 0
        disabled_count = 0
        summary = []
        
        for model_key, model_config in models_config.items():
            display_name = model_key.replace('_', ' ').title()
            is_enabled = model_config.get('enabled', True)
            
            if is_enabled:
                enabled_count += 1
                param_count = len(model_config.get('params', {}))
                summary.append(f"✅ {display_name} ({param_count} tunable params)")
            else:
                disabled_count += 1
                summary.append(f"❌ {display_name} (disabled)")
        
        result = f"Models Configuration: {enabled_count} enabled, {disabled_count} disabled\n"
        result += "\n".join(summary)
        return result
    
    def optimize_for_speed(self):
        """Optimize configuration for faster execution"""
        self.set('cross_validation.cv_folds', 3)
        self.set('hyperparameter_tuning.n_iter', 10)
        self.set('hyperparameter_tuning.cv_folds', 2)
        print("⚡ Configuration optimized for speed")
    
    def optimize_for_accuracy(self):
        """Optimize configuration for better accuracy"""
        self.set('cross_validation.cv_folds', 10)
        self.set('hyperparameter_tuning.n_iter', 50)
        self.set('hyperparameter_tuning.cv_folds', 5)
        print("🎯 Configuration optimized for accuracy")