import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import joblib

class EnsembleMethods:
    def __init__(self, base_models=None):
        self.base_models = base_models or {}
        self.ensemble_models = {}
        self.ensemble_results = {}
        
    def create_voting_classifier(self, voting='soft'):
        """Create voting classifier from base models"""
        if not self.base_models:
            raise ValueError("No base models provided")
        
        # Prepare estimators list
        estimators = [(name, clone(model)) for name, model in self.base_models.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting
        )
        
        self.ensemble_models['Voting_' + voting.title()] = voting_clf
        return voting_clf
    
    def create_stacking_classifier(self, meta_classifier=None, cv=5):
        """Create stacking classifier with meta-learner"""
        if not self.base_models:
            raise ValueError("No base models provided")
        
        if meta_classifier is None:
            meta_classifier = LogisticRegression(random_state=42)
        
        # Prepare estimators list
        estimators = [(name, clone(model)) for name, model in self.base_models.items()]
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=cv,
            n_jobs=-1
        )
        
        self.ensemble_models['Stacking'] = stacking_clf
        return stacking_clf
    
    def create_weighted_ensemble(self, weights=None):
        """Create weighted voting ensemble"""
        if weights is None:
            # Equal weights
            weights = [1.0] * len(self.base_models)
        
        estimators = [(name, clone(model)) for name, model in self.base_models.items()]
        
        weighted_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        
        self.ensemble_models['Weighted_Ensemble'] = weighted_ensemble
        return weighted_ensemble
    
    def train_all_ensembles(self, X_train, y_train):
        """Train all ensemble methods"""
        print("\n🔗 Training ensemble methods...")
        
        # Create ensemble methods
        self.create_voting_classifier(voting='soft')
        self.create_voting_classifier(voting='hard')
        self.create_stacking_classifier()
        
        # Train each ensemble
        for name, ensemble in self.ensemble_models.items():
            print(f"  → Training {name}...")
            ensemble.fit(X_train, y_train)
    
    def evaluate_ensembles(self, X_test, y_test, target_encoder):
        """Evaluate all ensemble methods"""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        for name, ensemble in self.ensemble_models.items():
            y_pred = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Convert back to original labels for report
            y_test_labels = target_encoder.inverse_transform(y_test)
            y_pred_labels = target_encoder.inverse_transform(y_pred)
            
            self.ensemble_results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test_labels, y_pred_labels),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
    
    def get_best_ensemble(self):
        """Get the best performing ensemble"""
        if not self.ensemble_results:
            return None, None
        
        best_name = max(self.ensemble_results.keys(), 
                       key=lambda x: self.ensemble_results[x]['accuracy'])
        return best_name, self.ensemble_models[best_name]
    
    def cross_validate_ensembles(self, X, y, cv=5):
        """Cross-validate ensemble methods"""
        cv_results = {}
        
        for name, ensemble in self.ensemble_models.items():
            print(f"Cross-validating {name}...")
            scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
        
        return cv_results
    
    def save_ensemble(self, ensemble_name, filepath):
        """Save trained ensemble model"""
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble {ensemble_name} not found")
        
        joblib.dump(self.ensemble_models[ensemble_name], filepath)
        print(f"Ensemble {ensemble_name} saved to {filepath}")
    
    def get_feature_importance_ensemble(self, ensemble_name):
        """Get feature importance for ensemble methods that support it"""
        if ensemble_name not in self.ensemble_models:
            return None
        
        ensemble = self.ensemble_models[ensemble_name]
        
        # For voting classifier, average the feature importances
        if hasattr(ensemble, 'estimators_'):
            importances = []
            for estimator in ensemble.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if importances:
                return np.mean(importances, axis=0)
        
        # For stacking classifier, use final estimator if it has feature importance
        if hasattr(ensemble, 'final_estimator_'):
            if hasattr(ensemble.final_estimator_, 'coef_'):
                return np.abs(ensemble.final_estimator_.coef_[0])
        
        return None