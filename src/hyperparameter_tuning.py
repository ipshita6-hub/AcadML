from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

class HyperparameterTuner:
    def __init__(self, cv_folds=3, n_jobs=-1, random_state=42):
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.tuned_models = {}
        self.best_params = {}
        self.tuning_results = {}
        
        # Define parameter grids for each model
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        
        # Base models
        self.base_models = {
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state),
            'KNN': KNeighborsClassifier()
        }
    
    def tune_hyperparameters(self, X_train, y_train, method='grid', n_iter=50):
        """Tune hyperparameters for all models"""
        print(f"\n🔧 Tuning hyperparameters using {method} search...")
        
        for model_name in self.base_models.keys():
            print(f"  → Tuning {model_name}...")
            
            base_model = self.base_models[model_name]
            param_grid = self.param_grids[model_name]
            
            if method == 'grid':
                search = GridSearchCV(
                    base_model, 
                    param_grid, 
                    cv=self.cv_folds,
                    scoring='accuracy',
                    n_jobs=self.n_jobs,
                    verbose=0
                )
            else:  # randomized search
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=n_iter,
                    cv=self.cv_folds,
                    scoring='accuracy',
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=0
                )
            
            # Fit the search
            search.fit(X_train, y_train)
            
            # Store results
            self.tuned_models[model_name] = search.best_estimator_
            self.best_params[model_name] = search.best_params_
            self.tuning_results[model_name] = {
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'cv_results': search.cv_results_
            }
    
    def print_tuning_results(self):
        """Print hyperparameter tuning results"""
        print("\n🎯 Hyperparameter Tuning Results:")
        print("=" * 80)
        
        for model_name, results in self.tuning_results.items():
            print(f"\n{model_name}:")
            print(f"  Best CV Score: {results['best_score']:.4f}")
            print(f"  Best Parameters:")
            for param, value in results['best_params'].items():
                print(f"    {param}: {value}")
    
    def get_tuning_summary(self):
        """Get a summary DataFrame of tuning results"""
        summary_data = []
        
        for model_name, results in self.tuning_results.items():
            summary_data.append({
                'Model': model_name,
                'Best_CV_Score': results['best_score'],
                'Num_Parameters_Tuned': len(results['best_params'])
            })
        
        return pd.DataFrame(summary_data).sort_values('Best_CV_Score', ascending=False)
    
    def get_best_model_tuned(self):
        """Get the best tuned model"""
        best_model_name = max(self.tuning_results.keys(), 
                             key=lambda x: self.tuning_results[x]['best_score'])
        return best_model_name, self.tuned_models[best_model_name]
    
    def compare_before_after(self, original_results, tuned_results):
        """Compare performance before and after tuning"""
        comparison_data = []
        
        for model_name in original_results.keys():
            if model_name in tuned_results:
                original_acc = original_results[model_name]['accuracy']
                tuned_acc = tuned_results[model_name]['accuracy']
                improvement = tuned_acc - original_acc
                
                comparison_data.append({
                    'Model': model_name,
                    'Original_Accuracy': original_acc,
                    'Tuned_Accuracy': tuned_acc,
                    'Improvement': improvement,
                    'Improvement_Percent': (improvement / original_acc) * 100
                })
        
        return pd.DataFrame(comparison_data).sort_values('Improvement', ascending=False)