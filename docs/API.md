# 📚 Academic Performance Prediction - API Documentation

## Overview

This document provides comprehensive API documentation for the Academic Performance Prediction ML pipeline.

## Core Modules

### 🔧 ConfigManager

Manages YAML-based configuration with validation and directory creation.

```python
from src.config_manager import ConfigManager

# Initialize with config file
config = ConfigManager('config.yaml')

# Get configuration values
sample_size = config.get('data.sample_size')
test_size = config.get('data.test_size', 0.2)  # with default

# Set configuration values
config.set('data.sample_size', 2000)

# Validate configuration
is_valid = config.validate_config()

# Save configuration
config.save_config('new_config.yaml')
```

**Methods:**
- `get(key_path, default=None)`: Get config value using dot notation
- `set(key_path, value)`: Set config value using dot notation
- `validate_config()`: Validate configuration values
- `save_config(path)`: Save current configuration to file
- `create_directories()`: Create necessary directories

### 📊 DataLoader

Handles data generation, loading, and preprocessing.

```python
from src.data_loader import DataLoader

# Initialize data loader
data_loader = DataLoader()

# Generate synthetic data
df = data_loader.generate_sample_data(n_samples=1000)

# Preprocess data
X, y, feature_names = data_loader.preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = data_loader.split_data(X, y, test_size=0.2)
```

**Methods:**
- `generate_sample_data(n_samples=1000)`: Generate synthetic academic data
- `load_data()`: Load data (generates sample if no path provided)
- `preprocess_data(df)`: Encode categorical variables and scale features
- `split_data(X, y, test_size=0.2, random_state=42)`: Split data into train/test sets

### 🤖 ModelTrainer

Trains and evaluates multiple ML models.

```python
from src.models import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Train all models
trainer.train_models(X_train, y_train)

# Evaluate models
trainer.evaluate_models(X_test, y_test, target_encoder)

# Get best model
best_name, best_model = trainer.get_best_model()

# Save model
trainer.save_model(best_name, best_model, 'model.pkl')
```

**Available Models:**
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)

**Methods:**
- `train_models(X_train, y_train)`: Train all models
- `evaluate_models(X_test, y_test, target_encoder)`: Evaluate trained models
- `get_best_model()`: Get best performing model
- `save_model(name, model, filepath)`: Save trained model

### 🔄 CrossValidator

Performs k-fold cross-validation with multiple metrics.

```python
from src.cross_validation import CrossValidator

# Initialize validator
cv = CrossValidator(cv_folds=5, random_state=42)

# Perform cross-validation
cv.perform_cross_validation(models, X, y)

# Print results
cv.print_cv_results()

# Get summary DataFrame
summary = cv.get_cv_summary()

# Get best model by CV
best_name, best_score = cv.get_best_model_cv()
```

**Metrics Calculated:**
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)

### 🎯 HyperparameterTuner

Optimizes model hyperparameters using grid or randomized search.

```python
from src.hyperparameter_tuning import HyperparameterTuner

# Initialize tuner
tuner = HyperparameterTuner(cv_folds=3, n_jobs=-1)

# Tune hyperparameters
tuner.tune_hyperparameters(X_train, y_train, method='randomized', n_iter=50)

# Print results
tuner.print_tuning_results()

# Get tuning summary
summary = tuner.get_tuning_summary()

# Get best tuned model
best_name, best_model = tuner.get_best_model_tuned()
```

**Search Methods:**
- `grid`: Exhaustive grid search
- `randomized`: Randomized parameter search

### 📈 ModelEvaluator

Comprehensive model evaluation with advanced metrics.

```python
from src.evaluation_metrics import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate all models
metrics = evaluator.evaluate_all_models(models, X_test, y_test, target_encoder)

# Print detailed results
evaluator.print_detailed_results()

# Get metrics comparison
comparison = evaluator.get_metrics_comparison()

# Get best models by metric
best_models = evaluator.get_best_models_by_metric()
```

**Metrics Included:**
- Accuracy & Balanced Accuracy
- Precision & Recall (macro/micro)
- F1-score (macro/micro)
- Matthews Correlation Coefficient
- Cohen's Kappa
- ROC-AUC (when applicable)

### 🔍 DataValidator

Validates data quality and identifies issues.

```python
from src.data_validation import DataValidator

# Initialize validator
validator = DataValidator()

# Validate dataset
results = validator.validate_dataset(df, target_column='performance')

# Print validation report
validator.print_validation_report()

# Get quality score
score = validator.get_data_quality_score()
```

**Validation Checks:**
- Missing values detection
- Duplicate rows identification
- Outlier detection (IQR method)
- Class balance analysis
- Feature correlation analysis
- Data distribution normality tests

### 🎨 EnhancedVisualizer

Creates interactive and static visualizations.

```python
from src.enhanced_visualizer import EnhancedVisualizer

# Initialize visualizer
viz = EnhancedVisualizer(results_dir='results')

# Create interactive data distribution
viz.plot_interactive_data_distribution(df)

# Create correlation heatmap
viz.plot_correlation_heatmap(df)

# Plot ROC curves
viz.plot_roc_curves(models, X_test, y_test, target_encoder)

# Create performance dashboard
viz.create_model_performance_dashboard(results)
```

**Visualization Types:**
- Interactive data distribution plots
- Correlation heatmaps
- ROC curves
- Precision-Recall curves
- Model comparison charts
- Performance dashboards

### 📋 ReportGenerator

Generates comprehensive ML pipeline reports.

```python
from src.report_generator import ReportGenerator

# Initialize generator
generator = ReportGenerator(config_manager)

# Generate comprehensive report
report = generator.generate_comprehensive_report(
    data_info=data_info,
    validation_results=validation_results,
    cv_results=cv_results,
    tuning_results=tuning_results,
    evaluation_results=evaluation_results,
    best_model_info=best_model_info
)

# Save report in multiple formats
files = generator.save_report(formats=['html', 'json', 'txt'])
```

**Report Formats:**
- HTML: Interactive web report
- JSON: Machine-readable format
- TXT: Plain text summary

### ⚡ PerformanceBenchmark

Benchmarks model training and prediction performance.

```python
from src.benchmarking import PerformanceBenchmark

# Initialize benchmark
benchmark = PerformanceBenchmark()

# Benchmark model training
training_benchmarks = benchmark.benchmark_model_training(models, X_train, y_train)

# Benchmark prediction speed
prediction_benchmarks = benchmark.benchmark_prediction_speed(models, X_test)

# Compare efficiency
efficiency = benchmark.compare_model_efficiency(training_benchmarks, prediction_benchmarks)

# Generate benchmark report
report = benchmark.generate_benchmark_report()
```

**Benchmark Metrics:**
- Training time and memory usage
- Prediction speed (predictions per second)
- Model size estimation
- Resource utilization

### 📝 MLLogger

Structured logging for ML pipeline operations.

```python
from src.logger import MLLogger

# Initialize logger
logger = MLLogger(config_manager)

# Log different types of events
logger.info("Starting model training")
logger.log_model_results(model_name, accuracy, metrics)
logger.log_cross_validation(model_name, cv_scores)
logger.log_hyperparameter_tuning(model_name, best_params, best_score)

# Session management
session_id = logger.create_session_log()
logger.end_session_log(session_id)
```

### 🖥️ CLIParser

Command-line interface for pipeline execution.

```python
from src.cli_parser import CLIParser

# Initialize parser
cli = CLIParser()

# Parse arguments
args = cli.parse_args()

# Validate arguments
cli.validate_args()

# Update configuration from CLI args
cli.update_config_from_args(config_manager)

# Check if component should run
should_run = cli.should_run_component('cross_validation')
```

**CLI Options:**
- Model selection (`--models rf gb svm`)
- Configuration overrides (`--samples 2000 --test-size 0.3`)
- Execution modes (`--dry-run`, `--profile`, `--quiet`)
- Output control (`--no-plots`, `--output-dir`)

## Usage Examples

### Basic Pipeline Execution

```python
# Simple execution with defaults
python main.py

# Custom configuration
python main.py --config custom_config.yaml

# Specific models only
python main.py --models rf gb --no-tuning

# Quiet mode with custom output
python main.py --quiet --output-dir results_v2
```

### Advanced Configuration

```python
# Performance profiling
python main.py --profile --models rf gb

# Dry run to validate configuration
python main.py --dry-run --config test_config.yaml

# Custom data and parameters
python main.py --data-file custom_data.csv --cv-folds 10 --tune-method grid
```

### Programmatic Usage

```python
from src.config_manager import ConfigManager
from src.data_loader import DataLoader
from src.models import ModelTrainer

# Load configuration
config = ConfigManager('config.yaml')

# Initialize components
data_loader = DataLoader()
trainer = ModelTrainer()

# Execute pipeline
df = data_loader.generate_sample_data(config.get('data.sample_size'))
X, y, features = data_loader.preprocess_data(df)
X_train, X_test, y_train, y_test = data_loader.split_data(X, y)

trainer.train_models(X_train, y_train)
trainer.evaluate_models(X_test, y_test, data_loader.target_encoder)

best_name, best_model = trainer.get_best_model()
```

## Error Handling

All modules include comprehensive error handling:

```python
try:
    # ML pipeline operations
    results = trainer.evaluate_models(X_test, y_test, target_encoder)
except Exception as e:
    logger.error(f"Model evaluation failed: {e}")
    # Graceful degradation or alternative approach
```

## Configuration Schema

See `config.yaml` for the complete configuration schema with all available options and their descriptions.

## Testing

Run the test suite:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test module
python -m pytest tests/test_models.py -v
```

## Performance Considerations

- Use `n_jobs=-1` for parallel processing where supported
- Consider data sampling for large datasets during development
- Enable profiling (`--profile`) to identify bottlenecks
- Use appropriate cross-validation folds based on dataset size

## Extending the Pipeline

To add new models:

1. Add model to `ModelTrainer.models` dictionary
2. Define parameter grid in `HyperparameterTuner.param_grids`
3. Update configuration schema if needed

To add new metrics:

1. Extend `ModelEvaluator.evaluate_model_comprehensive()`
2. Update report generation in `ReportGenerator`
3. Add visualization support if applicable