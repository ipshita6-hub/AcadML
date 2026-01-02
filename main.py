#!/usr/bin/env python3
"""
Academic Performance Prediction using Machine Learning Classification
"""

from src.data_loader import DataLoader
from src.models import ModelTrainer
from src.visualizer import Visualizer
import pandas as pd

def main():
    print("🎓 Academic Performance Prediction Project")
    print("=" * 50)
    
    # Initialize components
    data_loader = DataLoader()
    model_trainer = ModelTrainer()
    visualizer = Visualizer()
    
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
    
    # Visualize data distribution
    print("\n📊 Creating data visualizations...")
    visualizer.plot_data_distribution(df)
    
    # Preprocess data
    X, y, feature_names = data_loader.preprocess_data(df)
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    print("\n🤖 Training machine learning models...")
    model_trainer.train_models(X_train, y_train)
    
    # Evaluate models
    print("\n📊 Evaluating models...")
    model_trainer.evaluate_models(X_test, y_test, data_loader.target_encoder)
    
    # Visualize results
    print("\n📈 Creating result visualizations...")
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
    
    print("\n✅ Analysis complete! Check the 'results' folder for visualizations.")
    print("📁 Best model saved in 'models' folder.")

if __name__ == "__main__":
    main()