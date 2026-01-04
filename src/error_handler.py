"""Error handling and exception utilities for the ML pipeline"""

import functools
import logging
from typing import Callable, Any, Optional


class MLPipelineError(Exception):
    """Base exception for ML pipeline errors"""
    pass


class DataValidationError(MLPipelineError):
    """Raised when data validation fails"""
    pass


class ModelTrainingError(MLPipelineError):
    """Raised when model training fails"""
    pass


class PredictionError(MLPipelineError):
    """Raised when prediction fails"""
    pass


def handle_exceptions(logger: Optional[logging.Logger] = None, 
                     default_return: Any = None) -> Callable:
    """Decorator to handle exceptions in functions
    
    Args:
        logger: Logger instance for logging errors
        default_return: Default value to return on exception
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                if logger:
                    logger.error(error_msg)
                else:
                    print(f"❌ {error_msg}")
                return default_return
        return wrapper
    return decorator


def validate_input(func: Callable) -> Callable:
    """Decorator to validate function inputs
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if not args and not kwargs:
            raise ValueError(f"{func.__name__} requires at least one argument")
        return func(*args, **kwargs)
    return wrapper


class ErrorContext:
    """Context manager for error handling"""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """Initialize error context
        
        Args:
            operation_name: Name of the operation
            logger: Logger instance
        """
        self.operation_name = operation_name
        self.logger = logger
    
    def __enter__(self):
        """Enter context"""
        if self.logger:
            self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and handle exceptions"""
        if exc_type is not None:
            error_msg = f"Error in {self.operation_name}: {str(exc_val)}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"❌ {error_msg}")
            return False
        
        if self.logger:
            self.logger.info(f"Completed operation: {self.operation_name}")
        return True
