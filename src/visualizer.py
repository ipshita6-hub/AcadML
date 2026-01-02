import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os

class Visualizer:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        plt.style.use('default')
    
    def plot_data_distribution(self, df, save=True):
        """Plot distribution of features and target"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data Distribution Analysis', fontsize=16)
        
        # Numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for i, col in enumerate(numerical_cols[:4]):
            row, col_idx = i // 2, i % 2
            if row < 2 and col_idx < 2:
                axes[row, col_idx].hist(df[col], bins=30, alpha=0.7)
                axes[row, col_idx].set_title(f'Distribution of {col}')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('Frequency')
        
        # Target distribution
        axes[1, 2].pie(df['performance'].value_counts().values, 
                      labels=df['performance'].value_counts().index,
                      autopct='%1.1f%%')
        axes[1, 2].set_title('Performance Distribution')
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.results_dir}/data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results, save=True):
        """Plot model accuracy comparison"""
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color='skyblue', alpha=0.8)
        plt.title('Model Accuracy Comparison', fontsize=14)
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.results_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, results, target_encoder, save=True):
        """Plot confusion matrices for all models"""
        n_models = len(results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, result) in enumerate(results.items()):
            row, col = i // cols, i % cols
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=target_encoder.classes_,
                       yticklabels=target_encoder.classes_,
                       ax=axes[row, col])
            axes[row, col].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.results_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name, save=True):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importance - {model_name}')
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            
            if save:
                plt.savefig(f'{self.results_dir}/feature_importance_{model_name.replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print(f"Feature importance not available for {model_name}")