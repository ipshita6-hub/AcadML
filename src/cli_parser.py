import argparse
import sys
import os

class CLIParser:
    def __init__(self):
        self.parser = self.create_parser()
        self.args = None
    
    def create_parser(self):
        """Create command line argument parser"""
        parser = argparse.ArgumentParser(
            description='Academic Performance Prediction ML Pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py                                    # Run with default settings
  python main.py --config custom_config.yaml       # Use custom config
  python main.py --samples 2000 --test-size 0.3    # Override data settings
  python main.py --models rf gb --no-tuning        # Run only specific models
  python main.py --cv-folds 10 --tune-method grid  # Advanced ML settings
  python main.py --no-plots --quiet                # Minimal output mode
  python main.py --output-dir results_v2            # Custom output directory
            """
        )
        
        # Configuration
        config_group = parser.add_argument_group('Configuration')
        config_group.add_argument(
            '--config', '-c',
            type=str,
            default='config.yaml',
            help='Path to configuration file (default: config.yaml)'
        )
        config_group.add_argument(
            '--save-config',
            type=str,
            help='Save current configuration to specified file'
        )
        
        # Data settings
        data_group = parser.add_argument_group('Data Settings')
        data_group.add_argument(
            '--samples', '-n',
            type=int,
            help='Number of samples to generate (overrides config)'
        )
        data_group.add_argument(
            '--test-size',
            type=float,
            help='Test set size as fraction (0.0-1.0, overrides config)'
        )
        data_group.add_argument(
            '--random-state',
            type=int,
            help='Random state for reproducibility (overrides config)'
        )
        data_group.add_argument(
            '--data-file',
            type=str,
            help='Path to custom dataset CSV file'
        )
        
        # Model settings
        model_group = parser.add_argument_group('Model Settings')
        model_group.add_argument(
            '--models', '-m',
            nargs='+',
            choices=['rf', 'gb', 'svm', 'lr', 'knn', 'all'],
            help='Models to train (rf=Random Forest, gb=Gradient Boosting, svm=SVM, lr=Logistic Regression, knn=KNN)'
        )
        model_group.add_argument(
            '--cv-folds',
            type=int,
            help='Number of cross-validation folds (overrides config)'
        )
        model_group.add_argument(
            '--no-cv',
            action='store_true',
            help='Skip cross-validation'
        )
        
        # Hyperparameter tuning
        tuning_group = parser.add_argument_group('Hyperparameter Tuning')
        tuning_group.add_argument(
            '--no-tuning',
            action='store_true',
            help='Skip hyperparameter tuning'
        )
        tuning_group.add_argument(
            '--tune-method',
            choices=['grid', 'randomized'],
            help='Hyperparameter tuning method (overrides config)'
        )
        tuning_group.add_argument(
            '--tune-iter',
            type=int,
            help='Number of iterations for randomized search (overrides config)'
        )
        
        # Visualization settings
        viz_group = parser.add_argument_group('Visualization Settings')
        viz_group.add_argument(
            '--no-plots',
            action='store_true',
            help='Skip generating plots'
        )
        viz_group.add_argument(
            '--no-interactive',
            action='store_true',
            help='Skip interactive plots (HTML)'
        )
        viz_group.add_argument(
            '--plot-format',
            choices=['png', 'jpg', 'svg', 'pdf'],
            help='Output format for static plots (overrides config)'
        )
        
        # Output settings
        output_group = parser.add_argument_group('Output Settings')
        output_group.add_argument(
            '--output-dir', '-o',
            type=str,
            help='Output directory for results (overrides config)'
        )
        output_group.add_argument(
            '--model-dir',
            type=str,
            help='Directory to save trained models (overrides config)'
        )
        output_group.add_argument(
            '--no-save',
            action='store_true',
            help='Do not save models and results'
        )
        
        # Logging and verbosity
        log_group = parser.add_argument_group('Logging and Verbosity')
        log_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Reduce output verbosity'
        )
        log_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Increase output verbosity'
        )
        log_group.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            help='Set logging level (overrides config)'
        )
        log_group.add_argument(
            '--no-log-file',
            action='store_true',
            help='Disable logging to file'
        )
        
        # Validation and testing
        test_group = parser.add_argument_group('Validation and Testing')
        test_group.add_argument(
            '--no-validation',
            action='store_true',
            help='Skip data validation'
        )
        test_group.add_argument(
            '--run-tests',
            action='store_true',
            help='Run unit tests before execution'
        )
        
        # Utility options
        util_group = parser.add_argument_group('Utility Options')
        util_group.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be executed without running'
        )
        util_group.add_argument(
            '--profile',
            action='store_true',
            help='Enable performance profiling'
        )
        util_group.add_argument(
            '--version',
            action='version',
            version='Academic ML Pipeline v1.0.0'
        )
        
        return parser
    
    def parse_args(self, args=None):
        """Parse command line arguments"""
        self.args = self.parser.parse_args(args)
        return self.args
    
    def validate_args(self):
        """Validate parsed arguments"""
        if not self.args:
            raise ValueError("Arguments not parsed yet. Call parse_args() first.")
        
        errors = []
        
        # Validate test size
        if self.args.test_size is not None:
            if not 0 < self.args.test_size < 1:
                errors.append("--test-size must be between 0 and 1")
        
        # Validate sample count
        if self.args.samples is not None:
            if self.args.samples <= 0:
                errors.append("--samples must be positive")
        
        # Validate CV folds
        if self.args.cv_folds is not None:
            if self.args.cv_folds < 2:
                errors.append("--cv-folds must be at least 2")
        
        # Validate tune iterations
        if self.args.tune_iter is not None:
            if self.args.tune_iter <= 0:
                errors.append("--tune-iter must be positive")
        
        # Validate file paths
        if self.args.data_file and not os.path.exists(self.args.data_file):
            errors.append(f"Data file not found: {self.args.data_file}")
        
        if self.args.config and not os.path.exists(self.args.config):
            print(f"Warning: Config file not found: {self.args.config}. Using defaults.")
        
        # Check conflicting options
        if self.args.quiet and self.args.verbose:
            errors.append("Cannot use both --quiet and --verbose")
        
        if errors:
            print("Argument validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        return True
    
    def get_model_mapping(self):
        """Get mapping from CLI model codes to full names"""
        return {
            'rf': 'Random Forest',
            'gb': 'Gradient Boosting',
            'svm': 'SVM',
            'lr': 'Logistic Regression',
            'knn': 'KNN'
        }
    
    def get_selected_models(self):
        """Get list of selected model names"""
        if not self.args or not self.args.models:
            return None
        
        if 'all' in self.args.models:
            return list(self.get_model_mapping().values())
        
        mapping = self.get_model_mapping()
        return [mapping[code] for code in self.args.models if code in mapping]
    
    def update_config_from_args(self, config_manager):
        """Update configuration manager with CLI arguments"""
        if not self.args:
            return
        
        # Data settings
        if self.args.samples:
            config_manager.set('data.sample_size', self.args.samples)
        
        if self.args.test_size:
            config_manager.set('data.test_size', self.args.test_size)
        
        if self.args.random_state:
            config_manager.set('data.random_state', self.args.random_state)
        
        # Cross-validation settings
        if self.args.cv_folds:
            config_manager.set('cross_validation.cv_folds', self.args.cv_folds)
        
        # Hyperparameter tuning settings
        if self.args.tune_method:
            config_manager.set('hyperparameter_tuning.method', self.args.tune_method)
        
        if self.args.tune_iter:
            config_manager.set('hyperparameter_tuning.n_iter', self.args.tune_iter)
        
        # Visualization settings
        if self.args.no_plots:
            config_manager.set('visualization.save_plots', False)
        
        if self.args.no_interactive:
            config_manager.set('visualization.interactive_plots', False)
        
        if self.args.plot_format:
            config_manager.set('visualization.plot_format', self.args.plot_format)
        
        # Output settings
        if self.args.output_dir:
            config_manager.set('visualization.results_dir', self.args.output_dir)
        
        if self.args.model_dir:
            config_manager.set('output.model_dir', self.args.model_dir)
        
        if self.args.no_save:
            config_manager.set('output.save_models', False)
            config_manager.set('output.save_results', False)
        
        # Logging settings
        if self.args.log_level:
            config_manager.set('logging.level', self.args.log_level)
        
        if self.args.no_log_file:
            config_manager.set('logging.log_to_file', False)
        
        # Model selection
        selected_models = self.get_selected_models()
        if selected_models:
            # Disable all models first
            for model_key in ['random_forest', 'gradient_boosting', 'svm', 'logistic_regression', 'knn']:
                config_manager.set(f'models.{model_key}.enabled', False)
            
            # Enable selected models
            model_key_mapping = {
                'Random Forest': 'random_forest',
                'Gradient Boosting': 'gradient_boosting',
                'SVM': 'svm',
                'Logistic Regression': 'logistic_regression',
                'KNN': 'knn'
            }
            
            for model_name in selected_models:
                if model_name in model_key_mapping:
                    key = model_key_mapping[model_name]
                    config_manager.set(f'models.{key}.enabled', True)
    
    def print_args_summary(self):
        """Print summary of parsed arguments"""
        if not self.args:
            return
        
        print("\n⚙️  Command Line Arguments:")
        print("-" * 40)
        
        # Only show non-default values
        for arg, value in vars(self.args).items():
            if value is not None and value is not False:
                if isinstance(value, list):
                    value = ', '.join(value)
                print(f"  {arg.replace('_', '-')}: {value}")
    
    def should_run_component(self, component):
        """Check if a component should be run based on CLI args"""
        if not self.args:
            return True
        
        component_flags = {
            'cross_validation': not self.args.no_cv,
            'hyperparameter_tuning': not self.args.no_tuning,
            'visualization': not self.args.no_plots,
            'data_validation': not self.args.no_validation,
            'interactive_plots': not self.args.no_interactive
        }
        
        return component_flags.get(component, True)