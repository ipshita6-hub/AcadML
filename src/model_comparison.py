"""Model comparison and analysis utilities"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ModelComparator:
    """Compare and analyze multiple trained models"""
    
    def __init__(self):
        """Initialize model comparator"""
        self.comparison_results = {}
    
    def compare_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                      y_test: np.ndarray) -> pd.DataFrame:
        """Compare multiple models on test set
        
        Args:
            models: Dictionary of model_name -> model object
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            results.append(metrics)
            self.comparison_results[model_name] = metrics
        
        return pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    
    def get_best_model(self) -> Tuple[str, float]:
        """Get best model by accuracy
        
        Returns:
            Tuple of (model_name, accuracy)
        """
        if not self.comparison_results:
            return None, 0.0
        
        best_model = max(self.comparison_results.items(), 
                        key=lambda x: x[1]['Accuracy'])
        return best_model[0], best_model[1]['Accuracy']
    
    def get_worst_model(self) -> Tuple[str, float]:
        """Get worst model by accuracy
        
        Returns:
            Tuple of (model_name, accuracy)
        """
        if not self.comparison_results:
            return None, 0.0
        
        worst_model = min(self.comparison_results.items(), 
                         key=lambda x: x[1]['Accuracy'])
        return worst_model[0], worst_model[1]['Accuracy']
    
    def get_performance_gap(self) -> float:
        """Get performance gap between best and worst model
        
        Returns:
            Performance gap
        """
        if not self.comparison_results:
            return 0.0
        
        accuracies = [m['Accuracy'] for m in self.comparison_results.values()]
        return max(accuracies) - min(accuracies)
    
    def get_average_performance(self) -> Dict[str, float]:
        """Get average performance across all models
        
        Returns:
            Dictionary with average metrics
        """
        if not self.comparison_results:
            return {}
        
        metrics_list = list(self.comparison_results.values())
        
        avg_metrics = {
            'Accuracy': np.mean([m['Accuracy'] for m in metrics_list]),
            'Precision': np.mean([m['Precision'] for m in metrics_list]),
            'Recall': np.mean([m['Recall'] for m in metrics_list]),
            'F1-Score': np.mean([m['F1-Score'] for m in metrics_list])
        }
        
        return avg_metrics
    
    def get_model_consistency(self) -> Dict[str, float]:
        """Get consistency metrics for each model
        
        Returns:
            Dictionary with consistency scores
        """
        consistency = {}
        
        for model_name, metrics in self.comparison_results.items():
            # Calculate consistency as how close metrics are to each other
            metric_values = [metrics['Accuracy'], metrics['Precision'], 
                           metrics['Recall'], metrics['F1-Score']]
            consistency[model_name] = 1 - np.std(metric_values)
        
        return consistency
    
    def print_comparison_summary(self) -> None:
        """Print comparison summary"""
        if not self.comparison_results:
            print("No comparison results available")
            return
        
        print("\n🏆 Model Comparison Summary:")
        print("=" * 70)
        
        best_name, best_acc = self.get_best_model()
        worst_name, worst_acc = self.get_worst_model()
        avg_perf = self.get_average_performance()
        gap = self.get_performance_gap()
        
        print(f"\nBest Model: {best_name} (Accuracy: {best_acc:.4f})")
        print(f"Worst Model: {worst_name} (Accuracy: {worst_acc:.4f})")
        print(f"Performance Gap: {gap:.4f}")
        print(f"\nAverage Performance:")
        for metric, value in avg_perf.items():
            print(f"  {metric}: {value:.4f}")
        
        consistency = self.get_model_consistency()
        print(f"\nModel Consistency (Metric Alignment):")
        for model_name, score in sorted(consistency.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name}: {score:.4f}")
    
    def export_comparison(self, output_path: str = 'results/model_comparison.csv') -> str:
        """Export comparison results to CSV
        
        Args:
            output_path: Path to save CSV
            
        Returns:
            Path to saved file
        """
        if not self.comparison_results:
            print("No comparison results to export")
            return None
        
        df = pd.DataFrame(list(self.comparison_results.values()))
        df = df.sort_values('Accuracy', ascending=False)
        df.to_csv(output_path, index=False)
        
        print(f"✅ Comparison results exported to {output_path}")
        return output_path
    
    def get_model_recommendations(self) -> Dict[str, str]:
        """Get recommendations for model selection
        
        Returns:
            Dictionary with recommendations
        """
        if not self.comparison_results:
            return {}
        
        recommendations = {}
        
        # Best overall
        best_name, _ = self.get_best_model()
        recommendations['best_overall'] = best_name
        
        # Best precision
        best_precision = max(self.comparison_results.items(), 
                            key=lambda x: x[1]['Precision'])
        recommendations['best_precision'] = best_precision[0]
        
        # Best recall
        best_recall = max(self.comparison_results.items(), 
                         key=lambda x: x[1]['Recall'])
        recommendations['best_recall'] = best_recall[0]
        
        # Best F1
        best_f1 = max(self.comparison_results.items(), 
                     key=lambda x: x[1]['F1-Score'])
        recommendations['best_f1'] = best_f1[0]
        
        # Most consistent
        consistency = self.get_model_consistency()
        best_consistent = max(consistency.items(), key=lambda x: x[1])
        recommendations['most_consistent'] = best_consistent[0]
        
        return recommendations
