import time
import psutil
import numpy as np
import pandas as pd
from functools import wraps
import tracemalloc
import gc
from datetime import datetime
import json
import os

class PerformanceBenchmark:
    def __init__(self):
        self.benchmarks = {}
        self.system_info = self._get_system_info()
        self.memory_snapshots = {}
        
    def _get_system_info(self):
        """Get system information for benchmarking context"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'platform': psutil.os.name,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    
    def benchmark_function(self, name=None, track_memory=True):
        """Decorator to benchmark function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = name or func.__name__
                
                # Start benchmarking
                start_time = time.time()
                start_cpu = time.process_time()
                
                if track_memory:
                    tracemalloc.start()
                    gc.collect()  # Clean up before measurement
                    start_memory = psutil.Process().memory_info().rss
                
                # Execute function
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                
                # End benchmarking
                end_time = time.time()
                end_cpu = time.process_time()
                
                benchmark_data = {
                    'wall_time': end_time - start_time,
                    'cpu_time': end_cpu - start_cpu,
                    'success': success,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                }
                
                if track_memory:
                    end_memory = psutil.Process().memory_info().rss
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    benchmark_data.update({
                        'memory_used': end_memory - start_memory,
                        'memory_peak': peak,
                        'memory_current': current
                    })
                
                # Store benchmark
                if func_name not in self.benchmarks:
                    self.benchmarks[func_name] = []
                self.benchmarks[func_name].append(benchmark_data)
                
                return result
            return wrapper
        return decorator
    
    def benchmark_model_training(self, models, X_train, y_train):
        """Benchmark model training performance"""
        training_benchmarks = {}
        
        for model_name, model in models.items():
            print(f"Benchmarking {model_name} training...")
            
            # Multiple runs for statistical significance
            runs = []
            for run in range(3):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                # Train model
                model.fit(X_train, y_train)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                runs.append({
                    'run': run + 1,
                    'training_time': end_time - start_time,
                    'memory_used': end_memory - start_memory,
                    'model_size': self._estimate_model_size(model)
                })
            
            # Calculate statistics
            training_times = [r['training_time'] for r in runs]
            memory_usage = [r['memory_used'] for r in runs]
            
            training_benchmarks[model_name] = {
                'runs': runs,
                'avg_training_time': np.mean(training_times),
                'std_training_time': np.std(training_times),
                'avg_memory_usage': np.mean(memory_usage),
                'std_memory_usage': np.std(memory_usage),
                'min_training_time': np.min(training_times),
                'max_training_time': np.max(training_times)
            }
        
        return training_benchmarks
    
    def benchmark_prediction_speed(self, models, X_test, batch_sizes=[1, 10, 100]):
        """Benchmark prediction speed for different batch sizes"""
        prediction_benchmarks = {}
        
        for model_name, model in models.items():
            model_benchmarks = {}
            
            for batch_size in batch_sizes:
                if batch_size > len(X_test):
                    continue
                
                # Prepare batch
                X_batch = X_test[:batch_size]
                
                # Benchmark prediction
                times = []
                for _ in range(10):  # Multiple runs
                    start_time = time.time()
                    predictions = model.predict(X_batch)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                model_benchmarks[f'batch_{batch_size}'] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'predictions_per_second': batch_size / np.mean(times),
                    'time_per_prediction': np.mean(times) / batch_size
                }
            
            prediction_benchmarks[model_name] = model_benchmarks
        
        return prediction_benchmarks
    
    def benchmark_cross_validation(self, cv_validator, models, X, y):
        """Benchmark cross-validation performance"""
        cv_benchmarks = {}
        
        for model_name, model in models.items():
            print(f"Benchmarking {model_name} cross-validation...")
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            # Perform cross-validation (simplified version)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            cv_benchmarks[model_name] = {
                'cv_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'cv_scores': scores.tolist(),
                'avg_score': scores.mean(),
                'std_score': scores.std()
            }
        
        return cv_benchmarks
    
    def benchmark_hyperparameter_tuning(self, hp_tuner, model_name, model, X_train, y_train, param_grid):
        """Benchmark hyperparameter tuning performance"""
        print(f"Benchmarking {model_name} hyperparameter tuning...")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        from sklearn.model_selection import GridSearchCV
        
        # Simplified grid search for benchmarking
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=1)
        grid_search.fit(X_train, y_train)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        return {
            'tuning_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'n_combinations': len(grid_search.cv_results_['params']),
            'best_score': grid_search.best_score_,
            'time_per_combination': (end_time - start_time) / len(grid_search.cv_results_['params'])
        }
    
    def _estimate_model_size(self, model):
        """Estimate model size in bytes"""
        try:
            import pickle
            return len(pickle.dumps(model))
        except:
            return 0
    
    def get_resource_usage(self):
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': memory.used,
            'memory_available': memory.available,
            'timestamp': datetime.now().isoformat()
        }
    
    def monitor_resources(self, duration=60, interval=5):
        """Monitor system resources over time"""
        monitoring_data = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            monitoring_data.append(self.get_resource_usage())
            time.sleep(interval)
        
        return monitoring_data
    
    def compare_model_efficiency(self, training_benchmarks, prediction_benchmarks):
        """Compare model efficiency across different metrics"""
        efficiency_scores = {}
        
        for model_name in training_benchmarks.keys():
            # Normalize metrics (lower is better for time/memory, higher is better for speed)
            training_time = training_benchmarks[model_name]['avg_training_time']
            memory_usage = training_benchmarks[model_name]['avg_memory_usage']
            
            # Get prediction speed (predictions per second for batch_10)
            pred_speed = prediction_benchmarks.get(model_name, {}).get('batch_10', {}).get('predictions_per_second', 0)
            
            # Calculate efficiency score (normalized)
            efficiency_scores[model_name] = {
                'training_efficiency': 1 / (training_time + 1e-6),  # Higher is better
                'memory_efficiency': 1 / (memory_usage + 1e-6),    # Higher is better
                'prediction_speed': pred_speed,                     # Higher is better
                'overall_efficiency': (1 / (training_time + 1e-6)) * (pred_speed + 1e-6) / (memory_usage + 1e-6)
            }
        
        return efficiency_scores
    
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        report = {
            'system_info': self.system_info,
            'benchmark_summary': {},
            'detailed_benchmarks': self.benchmarks,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate summary statistics
        for func_name, runs in self.benchmarks.items():
            if runs:
                wall_times = [r['wall_time'] for r in runs if r['success']]
                cpu_times = [r['cpu_time'] for r in runs if r['success']]
                
                summary = {
                    'total_runs': len(runs),
                    'successful_runs': sum(1 for r in runs if r['success']),
                    'avg_wall_time': np.mean(wall_times) if wall_times else 0,
                    'avg_cpu_time': np.mean(cpu_times) if cpu_times else 0,
                    'success_rate': sum(1 for r in runs if r['success']) / len(runs)
                }
                
                if any('memory_used' in r for r in runs):
                    memory_usage = [r['memory_used'] for r in runs if 'memory_used' in r and r['success']]
                    summary['avg_memory_usage'] = np.mean(memory_usage) if memory_usage else 0
                
                report['benchmark_summary'][func_name] = summary
        
        return report
    
    def save_benchmark_report(self, output_dir='results'):
        """Save benchmark report to file"""
        report = self.generate_benchmark_report()
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = os.path.join(output_dir, f'benchmark_report_{timestamp_str}.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save as readable text
        txt_path = os.path.join(output_dir, f'benchmark_report_{timestamp_str}.txt')
        with open(txt_path, 'w') as f:
            f.write("PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("SYSTEM INFO:\n")
            f.write("-" * 20 + "\n")
            for key, value in report['system_info'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("BENCHMARK SUMMARY:\n")
            f.write("-" * 20 + "\n")
            for func_name, summary in report['benchmark_summary'].items():
                f.write(f"\n{func_name}:\n")
                f.write(f"  Runs: {summary['total_runs']}\n")
                f.write(f"  Success Rate: {summary['success_rate']:.2%}\n")
                f.write(f"  Avg Wall Time: {summary['avg_wall_time']:.4f}s\n")
                f.write(f"  Avg CPU Time: {summary['avg_cpu_time']:.4f}s\n")
                if 'avg_memory_usage' in summary:
                    f.write(f"  Avg Memory Usage: {summary['avg_memory_usage'] / 1024 / 1024:.2f} MB\n")
        
        return [json_path, txt_path]
    
    def print_benchmark_summary(self):
        """Print benchmark summary to console"""
        print("\n⚡ Performance Benchmark Summary")
        print("=" * 50)
        
        report = self.generate_benchmark_report()
        
        print(f"System: {report['system_info']['cpu_count']} CPUs, "
              f"{report['system_info']['memory_total'] / 1024**3:.1f} GB RAM")
        
        for func_name, summary in report['benchmark_summary'].items():
            print(f"\n{func_name}:")
            print(f"  ⏱️  Avg Time: {summary['avg_wall_time']:.4f}s")
            print(f"  ✅ Success Rate: {summary['success_rate']:.2%}")
            if 'avg_memory_usage' in summary:
                print(f"  💾 Avg Memory: {summary['avg_memory_usage'] / 1024 / 1024:.2f} MB")

import sys  # Add this import