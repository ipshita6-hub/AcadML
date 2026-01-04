import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceComparison:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.comparisons = {}
    
    def compare_models_detailed(self, evaluation_results):
        """Create detailed model comparison"""
        if not evaluation_results:
            return None
        
        comparison_data = []
        
        for model_name, metrics in evaluation_results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                row = {
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics.get('precision_macro', 0),
                    'Recall': metrics.get('recall_macro', 0),
                    'F1-Score': metrics.get('f1_macro', 0),
                    'Balanced Acc': metrics.get('balanced_accuracy', 0),
                    'Matthews CC': metrics.get('matthews_corrcoef', 0),
                    'Cohen Kappa': metrics.get('cohen_kappa', 0)
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        self.comparisons['detailed'] = comparison_df
        return comparison_df
    
    def get_performance_ranking(self, evaluation_results):
        """Get model ranking by different metrics"""
        comparison_df = self.compare_models_detailed(evaluation_results)
        
        if comparison_df is None:
            return None
        
        rankings = {}
        
        for metric in ['Accuracy', 'F1-Score', 'Balanced Acc', 'Matthews CC']:
            if metric in comparison_df.columns:
                ranking = comparison_df[['Model', metric]].sort_values(metric, ascending=False)
                ranking['Rank'] = range(1, len(ranking) + 1)
                rankings[metric] = ranking
        
        return rankings
    
    def get_performance_improvement(self, baseline_model, target_model, evaluation_results):
        """Calculate improvement from baseline to target model"""
        if not evaluation_results:
            return None
        
        baseline_metrics = evaluation_results.get(baseline_model, {})
        target_metrics = evaluation_results.get(target_model, {})
        
        if not baseline_metrics or not target_metrics:
            return None
        
        improvements = {}
        
        for metric in baseline_metrics.keys():
            if metric in target_metrics:
                baseline_val = baseline_metrics[metric]
                target_val = target_metrics[metric]
                
                if isinstance(baseline_val, (int, float)) and isinstance(target_val, (int, float)):
                    improvement = target_val - baseline_val
                    improvement_pct = (improvement / baseline_val * 100) if baseline_val != 0 else 0
                    
                    improvements[metric] = {
                        'baseline': baseline_val,
                        'target': target_val,
                        'improvement': improvement,
                        'improvement_percent': improvement_pct
                    }
        
        return improvements
    
    def plot_performance_comparison(self, evaluation_results, save=True):
        """Plot performance comparison across models"""
        comparison_df = self.compare_models_detailed(evaluation_results)
        
        if comparison_df is None:
            return
        
        # Select key metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        plot_data = comparison_df[['Model'] + metrics].set_index('Model')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data.plot(kind='bar', ax=ax, width=0.8)
        
        plt.title('Model Performance Comparison', fontsize=16, pad=20)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.results_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metric_distribution(self, evaluation_results, metric='Accuracy', save=True):
        """Plot distribution of a specific metric across models"""
        comparison_df = self.compare_models_detailed(evaluation_results)
        
        if comparison_df is None or metric not in comparison_df.columns:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(comparison_df))))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.title(f'{metric} Distribution Across Models', fontsize=14, pad=20)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, max(comparison_df[metric]) * 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.results_dir}/metric_distribution_{metric.lower()}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_report(self, evaluation_results, output_path='results/performance_comparison.txt'):
        """Generate text report of performance comparison"""
        comparison_df = self.compare_models_detailed(evaluation_results)
        rankings = self.get_performance_ranking(evaluation_results)
        
        if comparison_df is None:
            return None
        
        with open(output_path, 'w') as f:
            f.write("PERFORMANCE COMPARISON REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall rankings
            f.write("OVERALL RANKINGS (by Accuracy):\n")
            f.write("-" * 70 + "\n")
            for idx, row in comparison_df.iterrows():
                rank = idx + 1
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                f.write(f"{medal} {row['Model']:20s} - Accuracy: {row['Accuracy']:.4f}\n")
            
            f.write("\n")
            
            # Detailed metrics
            f.write("DETAILED METRICS:\n")
            f.write("-" * 70 + "\n")
            f.write(comparison_df.to_string(index=False))
            
            f.write("\n\n")
            
            # Rankings by metric
            if rankings:
                f.write("RANKINGS BY METRIC:\n")
                f.write("-" * 70 + "\n")
                for metric, ranking_df in rankings.items():
                    f.write(f"\n{metric}:\n")
                    for idx, row in ranking_df.iterrows():
                        f.write(f"  {int(row['Rank'])}. {row['Model']:20s} - {row[metric]:.4f}\n")
            
            f.write("\n")
            
            # Performance insights
            f.write("PERFORMANCE INSIGHTS:\n")
            f.write("-" * 70 + "\n")
            
            best_model = comparison_df.iloc[0]
            worst_model = comparison_df.iloc[-1]
            avg_accuracy = comparison_df['Accuracy'].mean()
            
            f.write(f"Best Model: {best_model['Model']} ({best_model['Accuracy']:.4f})\n")
            f.write(f"Worst Model: {worst_model['Model']} ({worst_model['Accuracy']:.4f})\n")
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Performance Spread: {best_model['Accuracy'] - worst_model['Accuracy']:.4f}\n")
        
        print(f"✅ Comparison report saved to {output_path}")
        return output_path