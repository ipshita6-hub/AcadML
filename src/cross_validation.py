import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class CrossValidator:
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_results = {}
        
    def perform_cross_validation(self, models, X, y):
        """Perform cross-validation on all models"""
        print(f"\n🔄 Performing {self.cv_folds}-fold cross-validation...")
        
        # Use stratified k-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            print(f"  → Cross-validating {name}...")
            
            # Perform cross-validation for multiple metrics
            cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision_macro')
            cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall_macro')
            cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
            
            self.cv_results[name] = {
                'accuracy_scores': cv_accuracy,
                'precision_scores': cv_precision,
                'recall_scores': cv_recall,
                'f1_scores': cv_f1,
                'accuracy_mean': cv_accuracy.mean(),
                'accuracy_std': cv_accuracy.std(),
                'precision_mean': cv_precision.mean(),
                'precision_std': cv_precision.std(),
                'recall_mean': cv_recall.mean(),
                'recall_std': cv_recall.std(),
                'f1_mean': cv_f1.mean(),
                'f1_std': cv_f1.std()
            }
    
    def get_cv_summary(self):
        """Get a summary DataFrame of cross-validation results"""
        summary_data = []
        
        for model_name, results in self.cv_results.items():
            summary_data.append({
                'Model': model_name,
                'CV_Accuracy_Mean': results['accuracy_mean'],
                'CV_Accuracy_Std': results['accuracy_std'],
                'CV_Precision_Mean': results['precision_mean'],
                'CV_Precision_Std': results['precision_std'],
                'CV_Recall_Mean': results['recall_mean'],
                'CV_Recall_Std': results['recall_std'],
                'CV_F1_Mean': results['f1_mean'],
                'CV_F1_Std': results['f1_std']
            })
        
        return pd.DataFrame(summary_data)
    
    def print_cv_results(self):
        """Print formatted cross-validation results"""
        print("\n📊 Cross-Validation Results:")
        print("=" * 80)
        
        for model_name, results in self.cv_results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {results['accuracy_mean']:.4f} (±{results['accuracy_std']:.4f})")
            print(f"  Precision: {results['precision_mean']:.4f} (±{results['precision_std']:.4f})")
            print(f"  Recall:    {results['recall_mean']:.4f} (±{results['recall_std']:.4f})")
            print(f"  F1-Score:  {results['f1_mean']:.4f} (±{results['f1_std']:.4f})")
    
    def get_best_model_cv(self):
        """Get the best model based on cross-validation accuracy"""
        best_model = max(self.cv_results.keys(), 
                        key=lambda x: self.cv_results[x]['accuracy_mean'])
        return best_model, self.cv_results[best_model]['accuracy_mean']