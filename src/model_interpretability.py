import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
import warnings

class ModelInterpreter:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.interpretation_results = {}
    
    def calculate_permutation_importance(self, model, X_test, y_test, feature_names, n_repeats=10):
        """Calculate permutation importance for any model"""
        print(f"Calculating permutation importance...")
        
        perm_importance = permutation_importance(
            model, X_test, y_test, 
            n_repeats=n_repeats, 
            random_state=42,
            scoring='accuracy'
        )
        
        # Create DataFrame for easier handling
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def plot_permutation_importance(self, importance_df, model_name, save=True):
        """Plot permutation importance"""
        plt.figure(figsize=(10, 8))
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=True)
        
        # Create horizontal bar plot with error bars
        plt.barh(range(len(importance_df)), importance_df['importance_mean'], 
                xerr=importance_df['importance_std'], alpha=0.7, color='skyblue')
        
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Permutation Importance')
        plt.title(f'Permutation Feature Importance - {model_name}')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (mean_imp, std_imp) in enumerate(zip(importance_df['importance_mean'], 
                                                   importance_df['importance_std'])):
            plt.text(mean_imp + std_imp + 0.001, i, f'{mean_imp:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.results_dir}/permutation_importance_{model_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_interactions(self, model, X_test, feature_names, top_features=5):
        """Analyze feature interactions for tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            print(f"Feature interaction analysis not available for this model type")
            return None
        
        # Get top features
        feature_importance = model.feature_importances_
        top_indices = np.argsort(feature_importance)[-top_features:]
        top_feature_names = [feature_names[i] for i in top_indices]
        
        print(f"Top {top_features} most important features:")
        for i, (idx, name) in enumerate(zip(top_indices, top_feature_names)):
            print(f"  {i+1}. {name}: {feature_importance[idx]:.4f}")
        
        return {
            'top_features': top_feature_names,
            'top_indices': top_indices,
            'importances': feature_importance[top_indices]
        }
    
    def generate_decision_rules(self, model, feature_names, max_depth=3):
        """Generate human-readable decision rules for tree-based models"""
        if hasattr(model, 'estimators_'):
            # For ensemble models, use the first tree
            if len(model.estimators_) > 0:
                tree = model.estimators_[0]
                if hasattr(tree, 'tree_'):
                    rules = export_text(tree, feature_names=feature_names, max_depth=max_depth)
                    return rules
        elif hasattr(model, 'tree_'):
            # For single decision tree
            rules = export_text(model, feature_names=feature_names, max_depth=max_depth)
            return rules
        
        return "Decision rules not available for this model type"
    
    def calculate_model_confidence(self, model, X_test):
        """Calculate prediction confidence scores"""
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)
            # Confidence is the maximum probability for each prediction
            confidence_scores = np.max(probabilities, axis=1)
            
            confidence_stats = {
                'mean_confidence': np.mean(confidence_scores),
                'std_confidence': np.std(confidence_scores),
                'min_confidence': np.min(confidence_scores),
                'max_confidence': np.max(confidence_scores),
                'low_confidence_count': np.sum(confidence_scores < 0.6),
                'high_confidence_count': np.sum(confidence_scores > 0.8)
            }
            
            return confidence_scores, confidence_stats
        
        return None, None
    
    def plot_confidence_distribution(self, confidence_scores, model_name, save=True):
        """Plot distribution of prediction confidence"""
        if confidence_scores is None:
            print("No confidence scores available for this model")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Histogram of confidence scores
        plt.hist(confidence_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidence_scores):.3f}')
        
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title(f'Prediction Confidence Distribution - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(f'{self.results_dir}/confidence_distribution_{model_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_prediction_errors(self, model, X_test, y_test, target_encoder):
        """Analyze prediction errors and misclassifications"""
        y_pred = model.predict(X_test)
        
        # Find misclassified samples
        misclassified_mask = y_pred != y_test
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return None
        
        # Analyze error patterns
        error_analysis = {
            'total_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(y_test),
            'error_by_class': {}
        }
        
        # Error analysis by true class
        for class_idx in np.unique(y_test):
            class_name = target_encoder.inverse_transform([class_idx])[0]
            class_mask = y_test == class_idx
            class_errors = np.sum(misclassified_mask & class_mask)
            class_total = np.sum(class_mask)
            
            error_analysis['error_by_class'][class_name] = {
                'errors': class_errors,
                'total': class_total,
                'error_rate': class_errors / class_total if class_total > 0 else 0
            }
        
        return error_analysis, misclassified_indices
    
    def generate_interpretation_report(self, model, model_name, X_test, y_test, 
                                     feature_names, target_encoder):
        """Generate comprehensive model interpretation report"""
        print(f"\n🔍 Generating interpretation report for {model_name}...")
        
        report = {
            'model_name': model_name,
            'interpretability_methods': []
        }
        
        # 1. Permutation Importance
        try:
            perm_importance = self.calculate_permutation_importance(
                model, X_test, y_test, feature_names
            )
            self.plot_permutation_importance(perm_importance, model_name)
            report['permutation_importance'] = perm_importance.to_dict('records')
            report['interpretability_methods'].append('Permutation Importance')
        except Exception as e:
            print(f"Could not calculate permutation importance: {e}")
        
        # 2. Feature Interactions (for tree-based models)
        try:
            interactions = self.analyze_feature_interactions(model, X_test, feature_names)
            if interactions:
                report['feature_interactions'] = interactions
                report['interpretability_methods'].append('Feature Interactions')
        except Exception as e:
            print(f"Could not analyze feature interactions: {e}")
        
        # 3. Decision Rules (for tree-based models)
        try:
            rules = self.generate_decision_rules(model, feature_names)
            if rules and "not available" not in rules:
                report['decision_rules'] = rules
                report['interpretability_methods'].append('Decision Rules')
        except Exception as e:
            print(f"Could not generate decision rules: {e}")
        
        # 4. Prediction Confidence
        try:
            confidence_scores, confidence_stats = self.calculate_model_confidence(model, X_test)
            if confidence_scores is not None:
                self.plot_confidence_distribution(confidence_scores, model_name)
                report['confidence_analysis'] = confidence_stats
                report['interpretability_methods'].append('Confidence Analysis')
        except Exception as e:
            print(f"Could not analyze prediction confidence: {e}")
        
        # 5. Error Analysis
        try:
            error_analysis, misclassified_indices = self.analyze_prediction_errors(
                model, X_test, y_test, target_encoder
            )
            if error_analysis:
                report['error_analysis'] = error_analysis
                report['interpretability_methods'].append('Error Analysis')
        except Exception as e:
            print(f"Could not analyze prediction errors: {e}")
        
        self.interpretation_results[model_name] = report
        return report
    
    def compare_model_interpretability(self):
        """Compare interpretability across models"""
        if not self.interpretation_results:
            print("No interpretation results available")
            return
        
        print("\n📊 Model Interpretability Comparison:")
        print("=" * 60)
        
        for model_name, report in self.interpretation_results.items():
            print(f"\n{model_name}:")
            print(f"  Available methods: {', '.join(report['interpretability_methods'])}")
            
            if 'confidence_analysis' in report:
                conf = report['confidence_analysis']
                print(f"  Mean confidence: {conf['mean_confidence']:.3f}")
                print(f"  Low confidence predictions: {conf['low_confidence_count']}")
            
            if 'error_analysis' in report:
                error = report['error_analysis']
                print(f"  Error rate: {error['error_rate']:.3f}")
    
    def save_interpretation_report(self, output_file=None):
        """Save interpretation results to file"""
        if not self.interpretation_results:
            print("No interpretation results to save")
            return
        
        if output_file is None:
            output_file = f"{self.results_dir}/model_interpretation_report.json"
        
        import json
        with open(output_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model_name, report in self.interpretation_results.items():
                serializable_report = {}
                for key, value in report.items():
                    if isinstance(value, np.ndarray):
                        serializable_report[key] = value.tolist()
                    else:
                        serializable_report[key] = value
                serializable_results[model_name] = serializable_report
            
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Interpretation report saved to {output_file}")
        return output_file