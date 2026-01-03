import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import export_text
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.explanations = {}
    
    def explain_model_predictions(self, model, X_test, y_test, feature_names, model_name):
        """Generate comprehensive model explanations"""
        print(f"🔍 Generating explanations for {model_name}...")
        
        explanations = {
            'model_name': model_name,
            'feature_importance': self._get_feature_importance(model, feature_names),
            'permutation_importance': self._get_permutation_importance(model, X_test, y_test, feature_names),
            'partial_dependence': self._get_partial_dependence(model, X_test, feature_names),
            'prediction_analysis': self._analyze_predictions(model, X_test, y_test, feature_names)
        }
        
        # Add model-specific explanations
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
            explanations['tree_rules'] = self._extract_tree_rules(model, feature_names)
        
        if hasattr(model, 'coef_'):
            explanations['linear_coefficients'] = self._analyze_linear_coefficients(model, feature_names)
        
        self.explanations[model_name] = explanations
        return explanations
    
    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'available': True,
                'importances': importance_df.to_dict('records'),
                'top_features': importance_df.head(5)['feature'].tolist()
            }
        else:
            return {'available': False, 'reason': 'Model does not provide feature importances'}
    
    def _get_permutation_importance(self, model, X_test, y_test, feature_names):
        """Calculate permutation importance"""
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            return {
                'available': True,
                'importances': importance_df.to_dict('records'),
                'top_features': importance_df.head(5)['feature'].tolist()
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _get_partial_dependence(self, model, X_test, feature_names):
        """Calculate partial dependence for top features"""
        try:
            # Get feature importance to select top features
            if hasattr(model, 'feature_importances_'):
                top_indices = np.argsort(model.feature_importances_)[-3:][::-1]
            else:
                top_indices = [0, 1, 2]  # Default to first 3 features
            
            pd_results = {}
            for idx in top_indices:
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                    try:
                        pd_data = partial_dependence(
                            model, X_test, features=[idx], kind='average'
                        )
                        pd_results[feature_name] = {
                            'values': pd_data['values'][0].tolist(),
                            'grid_values': pd_data['grid_values'][0].tolist()
                        }
                    except:
                        continue
            
            return {'available': True, 'partial_dependence': pd_results}
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _analyze_predictions(self, model, X_test, y_test, feature_names):
        """Analyze prediction patterns"""
        predictions = model.predict(X_test)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)
            confidence_scores = np.max(probabilities, axis=1)
        else:
            probabilities = None
            confidence_scores = None
        
        # Analyze correct vs incorrect predictions
        correct_mask = predictions == y_test
        accuracy = np.mean(correct_mask)
        
        analysis = {
            'accuracy': accuracy,
            'total_predictions': len(predictions),
            'correct_predictions': np.sum(correct_mask),
            'incorrect_predictions': np.sum(~correct_mask)
        }
        
        if confidence_scores is not None:
            analysis.update({
                'avg_confidence': np.mean(confidence_scores),
                'confidence_correct': np.mean(confidence_scores[correct_mask]),
                'confidence_incorrect': np.mean(confidence_scores[~correct_mask])
            })
        
        return analysis
    
    def _extract_tree_rules(self, model, feature_names):
        """Extract decision tree rules"""
        try:
            if hasattr(model, 'tree_'):
                # Single decision tree
                tree_rules = export_text(model, feature_names=feature_names, max_depth=3)
                return {'available': True, 'rules': tree_rules[:1000]}  # Limit length
            elif hasattr(model, 'estimators_'):
                # Ensemble of trees - extract from first few trees
                rules = []
                for i, tree in enumerate(model.estimators_[:3]):
                    if hasattr(tree, 'tree_'):
                        rule = export_text(tree, feature_names=feature_names, max_depth=2)
                        rules.append(f"Tree {i+1}:\n{rule[:500]}")
                return {'available': True, 'rules': '\n\n'.join(rules)}
            else:
                return {'available': False, 'reason': 'No tree structure found'}
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _analyze_linear_coefficients(self, model, feature_names):
        """Analyze linear model coefficients"""
        try:
  