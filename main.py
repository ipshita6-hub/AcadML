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
import pandas as pd

def main():
    print("🎓 Academic Performance Prediction Project - Enhanced Edition")
    print("=" * 60)
    
    # Initialize components
    data_loader = DataLoader()
    model_trainer = ModelTrainer()
    visualizer = Visualizer()
    enhanced_viz = EnhancedVisualizer()
    cv_validator = CrossValidator(cv_folds=5)
    hp_tuner = HyperparameterTuner(cv_folds=3)
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    df = data_loader.load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Show data info
    print("\n📈 Dataset Overview:")
    print(df.head())
    print(f"\nTarget distribution:")
    print(df['performance'].value_counts())
    
    # Enhanced visualizations
    print("\n🎨 Creating enhanced interactive visualizations...")
    
    # Interactive data distribution
    print("  → Interactive data distribution plots...")
    enhanced_viz.plot_interactive_data_distribution(df)
    
    # Correlation analysis
    print("  → Feature correlation heatmap...")
    enhanced_viz.plot_correlation_heatmap(df)
    
    # Preprocess data
    X, y, feature_names = data_loader.preprocess_data(df)
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    print("\n🤖 Training machine learning models...")
    model_trainer.train_models(X_train, y_train)
    
    # Perform cross-validation
    cv_validator.perform_cross_validation(model_trainer.models, X, y)
    cv_validator.print_cv_results()
    
    # Get CV summary
    cv_summary = cv_validator.get_cv_summary()
    print(f"\n📋 Cross-Validation Summary:")
    print(cv_summary.round(4))
    
    # Hyperparameter tuning
    hp_tuner.tune_hyperparameters(X_train, y_train, method='randomized', n_iter=20)
    hp_tuner.print_tuning_results()
    
    # Get tuning summary
    tuning_summary = hp_tuner.get_tuning_summary()
    print(f"\n🎯 Hyperparameter Tuning Summary:")
    print(tuning_summary)
    
    # Evaluate models
    print("\n📊 Evaluating models...")
    model_trainer.evaluate_models(X_test, y_test, data_loader.target_encoder)
    
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
    print(f"\n🏆 Best performing model: {best_name}")
    print(f"Best accuracy: {model_trainer.results[best_name]['accuracy']:.4f}")
    
    # Plot feature importance for best model
    visualizer.plot_feature_importance(best_model, feature_names, best_name)
    
    # Save best model
    model_trainer.save_model(best_name, best_model, f'models/best_model_{best_name.replace(" ", "_")}.pkl')
    
    print("\n✅ Enhanced analysis complete!")
    print("📁 Check the 'results' folder for:")
    print("   • Interactive HTML visualizations")
    print("   • High-resolution PNG images")
    print("   • ROC and Precision-Recall curves")
    print("   • Performance dashboard")
    print("   • Feature importance analysis")
    print("📁 Best model saved in 'models' folder.")

if __name__ == "__main__":
    main()