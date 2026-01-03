#!/usr/bin/env python3
"""
Academic Performance Prediction using Machine Learning Classification
Enhanced with Interactive Visualizations and Advanced Metrics
"""

from src.data_loader import DataLoader
from src.models import ModelTrainer
from src.visualizer import Visualizer
from src.enhanced_visualizer import EnhancedVisualizer
from src.cross_validation import CrossValidator
from src.hyperparameter_tuning import HyperparameterTuner
from src.evaluation_metrics import ModelEvaluator
from src.data_validation import DataValidator
from src.config_manager import ConfigManager
from src.logger import MLLogger
from src.cli_parser import CLIParser
import pandas as pd
import time
import sys

def main():
    start_time = time.time()
    
    # Parse command line arguments
    cli_parser = CLIParser()
    args = cli_parser.parse_args()
    cli_parser.validate_args()
    
    print("🎓 Academic Performance Prediction Project - Enhanced Edition")
    print("=" * 60)
    
    # Show CLI arguments if any were provided
    if len(sys.argv) > 1:
        cli_parser.print_args_summary()
    
    # Handle special modes
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Showing configuration without execution")
    
    if args.run_tests:
        print("\n🧪 Running unit tests...")
        import subprocess
        result = subprocess.run([sys.executable, 'tests/run_tests.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Tests failed. Exiting.")
            sys.exit(1)
        print("✅ All tests passed!")
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Update config with CLI arguments
    cli_parser.update_config_from_args(config)
    
    config.validate_config()
    config.create_directories()
    
    # Initialize logger
    logger = MLLogger(config)
    session_id = logger.create_session_log()
    
    # Save config if requested
    if args.save_config:
        config.save_config(args.save_config)
        print(f"✅ Configuration saved to {args.save_config}")
    
    if not args.quiet:
        config.print_config()
    
    if args.dry_run:
        print("\n✅ Dry run completed. Configuration validated.")
        return
    
    # Initialize components with configuration
    data_loader = DataLoader()
    model_trainer = ModelTrainer()
    visualizer = Visualizer(results_dir=config.get('visualization.results_dir'))
    enhanced_viz = EnhancedVisualizer(results_dir=config.get('visualization.results_dir'))
    cv_validator = CrossValidator(
        cv_folds=config.get('cross_validation.cv_folds'),
        random_state=config.get('cross_validation.random_state')
    )
    hp_tuner = HyperparameterTuner(
        cv_folds=config.get('hyperparameter_tuning.cv_folds'),
        n_jobs=config.get('hyperparameter_tuning.n_jobs'),
        random_state=config.get('hyperparameter_tuning.random_state')
    )
    evaluator = ModelEvaluator()
    validator = DataValidator()
    
    # Load and preprocess data
    if not args.quiet:
        print("\n📊 Loading and preprocessing data...")
    logger.info("Starting data loading and preprocessing")
    
    if args.data_file:
        # Load custom dataset
        try:
            df = pd.read_csv(args.data_file)
            logger.info(f"Loaded custom dataset from {args.data_file}")
        except Exception as e:
            logger.error(f"Failed to load custom dataset: {e}")
            sys.exit(1)
    else:
        # Generate sample data
        df = data_loader.generate_sample_data(n_samples=config.get('data.sample_size'))
    
    logger.log_data_info(df)
    
    if not args.quiet:
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
    
    # Validate data quality
    if cli_parser.should_run_component('data_validation'):
        logger.info("Starting data validation")
        validator.validate_dataset(df, target_column=config.get('data.target_column'))
        if not args.quiet:
            validator.print_validation_report()
        quality_score = validator.get_data_quality_score()
        logger.log_data_validation(quality_score)
        if not args.quiet:
            print(f"\n🏆 Data Quality Score: {quality_score:.1f}/100")
    
    # Show data info
    print("\n📈 Dataset Overview:")
    print(df.head())
    print(f"\nTarget distribution:")
    print(df['performance'].value_counts())
    
    # Enhanced visualizations
    if cli_parser.should_run_component('visualization'):
        if not args.quiet:
            print("\n🎨 Creating enhanced interactive visualizations...")
        
        # Interactive data distribution
        if not args.quiet:
            print("  → Interactive data distribution plots...")
        enhanced_viz.plot_interactive_data_distribution(df)
        
        # Correlation analysis
        if not args.quiet:
            print("  → Feature correlation heatmap...")
        enhanced_viz.plot_correlation_heatmap(df)
    
    # Preprocess data
    X, y, feature_names = data_loader.preprocess_data(df)
    X_train, X_test, y_train, y_test = data_loader.split_data(
        X, y, 
        test_size=config.get('data.test_size'),
        random_state=config.get('data.random_state')
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    print("\n🤖 Training machine learning models...")
    logger.info("Starting model training")
    training_start = time.time()
    
    model_trainer.train_models(X_train, y_train)
    
    training_end = time.time()
    logger.log_execution_time("Model training", training_start, training_end)
    
    # Perform cross-validation
    if cli_parser.should_run_component('cross_validation'):
        logger.info("Starting cross-validation")
        cv_validator.perform_cross_validation(model_trainer.models, X, y)
        if not args.quiet:
            cv_validator.print_cv_results()
        
        # Get CV summary
        cv_summary = cv_validator.get_cv_summary()
        if not args.quiet:
            print(f"\n📋 Cross-Validation Summary:")
            print(cv_summary.round(4))
    
    # Hyperparameter tuning
    if cli_parser.should_run_component('hyperparameter_tuning'):
        logger.info("Starting hyperparameter tuning")
        tuning_start = time.time()
        
        hp_tuner.tune_hyperparameters(
            X_train, y_train, 
            method=config.get('hyperparameter_tuning.method'),
            n_iter=config.get('hyperparameter_tuning.n_iter')
        )
        if not args.quiet:
            hp_tuner.print_tuning_results()
        
        tuning_end = time.time()
        logger.log_execution_time("Hyperparameter tuning", tuning_start, tuning_end)
        
        # Get tuning summary
        tuning_summary = hp_tuner.get_tuning_summary()
        if not args.quiet:
            print(f"\n🎯 Hyperparameter Tuning Summary:")
            print(tuning_summary)
    
    # Evaluate models
    print("\n📊 Evaluating models...")
    model_trainer.evaluate_models(X_test, y_test, data_loader.target_encoder)
    
    # Comprehensive evaluation with detailed metrics
    detailed_metrics = evaluator.evaluate_all_models(
        model_trainer.trained_models, X_test, y_test, data_loader.target_encoder
    )
    evaluator.print_detailed_results()
    
    # Get metrics comparison
    metrics_comparison = evaluator.get_metrics_comparison()
    print(f"\n📋 Comprehensive Metrics Comparison:")
    print(metrics_comparison)
    
    # Enhanced result visualizations
    print("\n📈 Creating enhanced result visualizations...")
    
    # Interactive model comparison
    print("  → Interactive model comparison...")
    enhanced_viz.plot_interactive_model_comparison(model_trainer.results)
    
    # ROC Curves
    print("  → ROC curves analysis...")
    enhanced_viz.plot_roc_curves(model_trainer.trained_models, X_test, y_test, data_loader.target_encoder)
    
    # Precision-Recall Curves
    print("  → Precision-Recall curves...")
    enhanced_viz.plot_precision_recall_curves(model_trainer.trained_models, X_test, y_test, data_loader.target_encoder)
    
    # Enhanced confusion matrices
    print("  → Enhanced confusion matrices...")
    enhanced_viz.plot_enhanced_confusion_matrices(model_trainer.results, data_loader.target_encoder)
    
    # Feature importance comparison
    print("  → Feature importance comparison...")
    enhanced_viz.plot_feature_importance_comparison(model_trainer.trained_models, feature_names)
    
    # Performance dashboard
    print("  → Performance dashboard...")
    enhanced_viz.create_model_performance_dashboard(model_trainer.results)
    
    # Original visualizations (for comparison)
    print("\n📊 Creating traditional visualizations...")
    visualizer.plot_data_distribution(df)
    visualizer.plot_model_comparison(model_trainer.results)
    visualizer.plot_confusion_matrices(model_trainer.results, data_loader.target_encoder)
    
    # Get best model and show feature importance
    best_name, best_model = model_trainer.get_best_model()
    logger.log_best_model(best_name, model_trainer.results[best_name]['accuracy'])
    
    print(f"\n🏆 Best performing model: {best_name}")
    print(f"Best accuracy: {model_trainer.results[best_name]['accuracy']:.4f}")
    
    # Plot feature importance for best model
    visualizer.plot_feature_importance(best_model, feature_names, best_name)
    
    # Save best model
    model_path = f'models/best_model_{best_name.replace(" ", "_")}.pkl'
    model_trainer.save_model(best_name, best_model, model_path)
    logger.info(f"Best model saved to {model_path}")
    
    # End session
    end_time = time.time()
    logger.log_execution_time("Total execution", start_time, end_time)
    logger.end_session_log(session_id)
    
    print("\n✅ Enhanced analysis complete!")
    print("📁 Check the 'results' folder for:")
    print("   • Interactive HTML visualizations")
    print("   • High-resolution PNG images")
    print("   • ROC and Precision-Recall curves")
    print("   • Performance dashboard")
    print("   • Feature importance analysis")
    print("📁 Best model saved in 'models' folder.")
    print("📋 Detailed logs saved in 'logs' folder.")

if __name__ == "__main__":
    main()