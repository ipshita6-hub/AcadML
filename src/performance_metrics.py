"""Performance metrics and monitoring utilities"""

import time
from typing import Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_used: float = 0.0
    status: str = "completed"
    
    def __str__(self) -> str:
        """String representation of metrics"""
        return (f"{self.operation_name}: {self.duration:.2f}s "
                f"(Memory: {self.memory_used:.2f}MB)")


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics: Dict[str, list] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation_name: str) -> None:
        """Start timing an operation
        
        Args:
            operation_name: Name of the operation
        """
        self.start_times[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return duration
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Duration in seconds
        """
        if operation_name not in self.start_times:
            raise ValueError(f"Timer for '{operation_name}' was not started")
        
        duration = time.time() - self.start_times[operation_name]
        
        if operation_name not in self.metrics:
            self.metrics[operation_name] = []
        
        self.metrics[operation_name].append(duration)
        del self.start_times[operation_name]
        
        return duration
    
    def get_average_time(self, operation_name: str) -> float:
        """Get average execution time for an operation
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Average duration in seconds
        """
        if operation_name not in self.metrics or not self.metrics[operation_name]:
            return 0.0
        
        times = self.metrics[operation_name]
        return sum(times) / len(times)
    
    def get_total_time(self, operation_name: str) -> float:
        """Get total execution time for an operation
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Total duration in seconds
        """
        if operation_name not in self.metrics:
            return 0.0
        
        return sum(self.metrics[operation_name])
    
    def get_call_count(self, operation_name: str) -> int:
        """Get number of times an operation was called
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Number of calls
        """
        if operation_name not in self.metrics:
            return 0
        
        return len(self.metrics[operation_name])
    
    def print_summary(self) -> None:
        """Print performance summary"""
        print("\n⏱️  Performance Summary:")
        print("=" * 60)
        
        for operation_name, times in self.metrics.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            call_count = len(times)
            
            print(f"\n{operation_name}:")
            print(f"  Calls: {call_count}")
            print(f"  Average: {avg_time:.4f}s")
            print(f"  Total: {total_time:.4f}s")
            print(f"  Min: {min(times):.4f}s | Max: {max(times):.4f}s")
    
    def get_summary_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary as dictionary
        
        Returns:
            Dictionary with performance metrics
        """
        summary = {}
        
        for operation_name, times in self.metrics.items():
            summary[operation_name] = {
                'calls': len(times),
                'average': sum(times) / len(times),
                'total': sum(times),
                'min': min(times),
                'max': max(times)
            }
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()


def time_operation(operation_name: str) -> Callable:
    """Decorator to time function execution
    
    Args:
        operation_name: Name of the operation
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                print(f"⏱️  {operation_name} completed in {duration:.4f}s")
        return wrapper
    return decorator
