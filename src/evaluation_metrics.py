from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, classification_report, confusion_matrix
)
import pandas as pd
import numpy as np

class ModelEvaluator:
    def __init__(self):
        self.detailed_results = {}
        self.metric_names = [
            'accuracy', 'balanced_accuracy', 'precision_macro', 'precision_micro',
            'recall_macro', 'recall_micro', 'f1_macro', 'f1_micro',
            'matthews_corrcoef', 'cohen_kappa'
        ]
    
    def evaluate_model_comprehensive(self, model, X_test, y_test, model_name, target_encoder):
        """Comprehensive evaluation of a single model"""
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        
        # Calculate all metrics
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
        
        # Precision, Recall, F1 (macro and micro averages)
        metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_test, y_pred, average='micro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_test, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_test, y_pred, average='micro', zero_division=0)
        
        # Advanced metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_test, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
        
        # ROC AUC (if probabilities available and multiclass)
        if y_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:
                    # Binary classification
                    if y_proba.ndim > 1:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                else:
                    # Multiclass classification
                    metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro')
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC for {model_name}: {e}")
        
        # Store detailed results
        self.detailed_results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                target_encoder.inverse_transform(y_test),
                target_encoder.inverse_transform(y_pred),
                output_dict=True
            )
        }
        
        return metrics
    
    def evaluate_all_models(self, models, X_test, y_test, target_encoder):
        """Evaluate all models comprehensively"""
        print("\n📊 Comprehensive Model Evaluation...")
        
        all_metrics = {}
        for model_name, model in models.items():
            print(f"  → Evaluating {model_name}...")
            metrics = self.evaluate_model_comprehensive(
                model, X_test, y_test, model_name, target_encoder
            )
            all_metrics[model_name] = metrics
        
        return all_metrics
    
    def get_metrics_comparison(self):
        """Get a comprehensive comparison of all metrics"""
        if not self.detailed_results:
            return None
        
        comparison_data = []
        for model_name, results in self.detailed_results.items():
            row = {'Model': model_name}
            row.update(results['metrics'])
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.round(4)
    
    def print_detailed_results(self):
        """Print detailed evaluation results"""
        print("\n📈 Detailed Model Evaluation Results:")
        print("=" * 100)
        
        for model_name, results in self.detailed_results.items():
            print(f"\n{model_name}:")
            print("-" * 50)
            
            metrics = results['metrics']
            print(f"Accuracy:           {metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
            print(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
            print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
            print(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
            print(f"Matthews Corr Coef: {metrics['matthews_corrcoef']:.4f}")
            print(f"Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")
            
            if 'roc_auc' in metrics:
                print(f"ROC AUC:            {metrics['roc_auc']:.4f}")
            if 'roc_auc_ovr' in metrics:
                print(f"ROC AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")
                print(f"ROC AUC (OvO):      {metrics['roc_auc_ovo']:.4f}")
    
    def get_best_models_by_metric(self):
        """Get best model for each metric"""
        if not self.detailed_results:
            return None
        
        best_models = {}
        metrics_df = self.get_metrics_comparison()
        
        for metric in self.metric_names:
            if metric in metrics_df.columns:
                best_idx = metrics_df[metric].idxmax()
                best_models[metric] = {
                    'model': metrics_df.loc[best_idx, 'Model'],
                    'score': metrics_df.loc[best_idx, metric]
                }
        
        return best_models
    
    def plot_metrics_heatmap(self, results, save=True):
        """Create heatmap of all metrics across models"""
        metrics_df = self.get_metrics_comparison()
        
        if metrics_df is None or metrics_df.empty:
            print("No metrics data available for heatmap")
            return
        
        # Select numeric columns only
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        heatmap_data = metrics_df[numeric_cols].set_index('Model') if 'Model' in metrics_df.columns else metrics_df[numeric_cols]
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, cbar_kws={'label': 'Score'}, linewidths=0.5)
        plt.title('Comprehensive Model Metrics Heatmap', fontsize=16, pad=20)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.results_dir}/metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_radar(self, results, model_name=None, save=True):
        """Create radar chart for model metrics"""
        if not self.detailed_results:
            print("No detailed results available")
            return
        
        # Select model
        if model_name is None:
            model_name = list(self.detailed_results.keys())[0]
        
        if model_name not in self.detailed_results:
            print(f"Model {model_name} not found")
            return
        
        metrics = self.detailed_results[model_name]['metrics']
        
        # Select key metrics for radar
        key_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'balanced_accuracy']
        values = [metrics.get(m, 0) for m in key_metrics]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color='#1f77b4')
        ax.fill(angles, values, alpha=0.25, color='#1f77b4')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(key_metrics, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'Model Performance Radar - {model_name}', fontsize=14, pad=20)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.results_dir}/metrics_radar_{model_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()

    def export_comparison_to_csv(self, output_path='results/model_comparison.csv'):
        """Export model comparison to CSV"""
        comparison_df = self.get_metrics_comparison()
        if comparison_df is not None:
            comparison_df.to_csv(output_path, index=False)
            print(f"✅ Model comparison exported to {output_path}")
            return output_path
        return None
    
    def export_detailed_results(self, output_dir='results'):
        """Export detailed results for each model"""
        import json
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, results in self.detailed_results.items():
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = {
                'metrics': results['metrics'],
                'classification_report': results['classification_report'],
                'confusion_matrix': results['confusion_matrix'].tolist()
            }
            
            filename = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_results.json')
            with open(filename, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            print(f"✅ Results exported for {model_name}")
        
        return output_dir