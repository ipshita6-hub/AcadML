import logging
import os
from datetime import datetime
import sys

class MLLogger:
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.logger = None
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logging configuration"""
        # Get configuration values
        if self.config:
            log_level = self.config.get('logging.level', 'INFO')
            log_to_file = self.config.get('logging.log_to_file', True)
            log_file = self.config.get('logging.log_file', 'logs/academic_ml.log')
            log_format = self.config.get('logging.log_format', 
                                       '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        else:
            log_level = 'INFO'
            log_to_file = True
            log_file = 'logs/academic_ml.log'
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logger
        self.logger = logging.getLogger('AcademicML')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            # Create log directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
        self.logger.info("Logger initialized successfully")
    
    def info(self, message: str) -> None:
        """Log info message
        
        Args:
            message: The message to log at INFO level
        """
        if self.logger:
            self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message
        
        Args:
            message: The message to log at DEBUG level
        """
        if self.logger:
            self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message
        
        Args:
            message: The message to log at WARNING level
        """
        if self.logger:
            self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message
        
        Args:
            message: The message to log at ERROR level
        """
        if self.logger:
            self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message
        
        Args:
            message: The message to log at CRITICAL level
        """
        if self.logger:
            self.logger.critical(message)
    
    def log_data_info(self, df):
        """Log dataset information"""
        self.info(f"Dataset loaded with shape: {df.shape}")
        self.info(f"Features: {list(df.columns)}")
        self.info(f"Target distribution: {df['performance'].value_counts().to_dict()}")
        self.info(f"Missing values: {df.isnull().sum().sum()}")
    
    def log_model_training(self, model_name):
        """Log model training start"""
        self.info(f"Starting training for {model_name}")
    
    def log_model_results(self, model_name, accuracy, metrics=None):
        """Log model results"""
        self.info(f"{model_name} - Accuracy: {accuracy:.4f}")
        if metrics:
            for metric, value in metrics.items():
                self.debug(f"{model_name} - {metric}: {value:.4f}")
    
    def log_cross_validation(self, model_name, cv_scores):
        """Log cross-validation results"""
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        self.info(f"{model_name} CV - Mean: {mean_score:.4f} (±{std_score:.4f})")
    
    def log_hyperparameter_tuning(self, model_name, best_params, best_score):
        """Log hyperparameter tuning results"""
        self.info(f"{model_name} - Best CV Score: {best_score:.4f}")
        self.debug(f"{model_name} - Best Parameters: {best_params}")
    
    def log_data_validation(self, quality_score, issues=None):
        """Log data validation results"""
        self.info(f"Data quality score: {quality_score:.1f}/100")
        if issues:
            for issue in issues:
                self.warning(f"Data validation issue: {issue}")
    
    def log_visualization_creation(self, plot_name, save_path=None):
        """Log visualization creation"""
        self.debug(f"Created visualization: {plot_name}")
        if save_path:
            self.debug(f"Saved to: {save_path}")
    
    def log_execution_time(self, operation, start_time, end_time):
        """Log execution time for operations"""
        duration = end_time - start_time
        self.info(f"{operation} completed in {duration:.2f} seconds")
    
    def log_best_model(self, model_name, score, metric='accuracy'):
        """Log best model selection"""
        self.info(f"Best model selected: {model_name} ({metric}: {score:.4f})")
    
    def log_error_with_context(self, error, context=""):
        """Log error with additional context"""
        error_msg = f"Error occurred: {str(error)}"
        if context:
            error_msg += f" | Context: {context}"
        self.error(error_msg)
    
    def create_session_log(self):
        """Create a new session log entry"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.info(f"=== New ML Session Started: {session_id} ===")
        return session_id
    
    def end_session_log(self, session_id):
        """End session log entry"""
        self.info(f"=== ML Session Ended: {session_id} ===")