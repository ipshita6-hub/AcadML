import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path

class ReportGenerator:
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.report_data = {}
        self.timestamp = datetime.now()
        
    def generate_comprehensive_report(self, data_info, validation_results, 
                                    cv_results, tuning_results, evaluation_results,
                                    best_model_info, execution_times=None):
        """Generate comprehensive ML pipeline report"""
        
        self.report_data = {
            'metadata': self._generate_metadata(),
            'data_summary': self._generate_data_summary(data_info, validation_results),
            'model_performance': self._generate_model_performance(evaluation_results),
            'cross_validation': self._generate_cv_summary(cv_results),
            'hyperparameter_tuning': self._generate_tuning_summary(tuning_results),
            'best_model': self._generate_best_model_summary(best_model_info),
            'recommendations': self._generate_recommendations(evaluation_results, validation_results),
            'execution_summary': self._generate_execution_summary(execution_times)
        }
        
        return self.report_data
    
    def _generate_metadata(self):
        """Generate report metadata"""
        return {
            'report_generated': self.timestamp.isoformat(),
            'pipeline_version': '1.0.0',
            'configuration': self.config.config if self.config else {},
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'libraries': {
                'pandas': pd.__version__,
                'numpy': np.__version__,
                'sklearn': '1.3.0+'  # Approximate version
            }
        }
    
    def _generate_data_summary(self, data_info, validation_results):
        """Generate data summary section"""
        summary = {
            'dataset_shape': data_info.get('shape', 'Unknown'),
            'features': data_info.get('features', []),
            'target_distribution': data_info.get('target_distribution', {}),
            'data_quality_score': validation_results.get('quality_score', 0) if validation_results else 0
        }
        
        if validation_results:
            summary.update({
                'missing_values': validation_results.get('missing_values', {}).get('total_missing', 0),
                'duplicate_rows': validation_results.get('duplicates', {}).get('duplicate_count', 0),
                'outliers_detected': sum(
                    info.get('outlier_count', 0) 
                    for info in validation_results.get('outliers', {}).values()
                ),
                'class_balance': validation_results.get('class_balance', {})
            })
        
        return summary
    
    def _generate_model_performance(self, evaluation_results):
        """Generate model performance comparison"""
        if not evaluation_results:
            return {}
        
        performance_data = []
        for model_name, metrics in evaluation_results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                performance_data.append({
                    'model': model_name,
                    'accuracy': metrics['accuracy'],
                    'precision_macro': metrics.get('precision_macro', 0),
                    'recall_macro': metrics.get('recall_macro', 0),
                    'f1_macro': metrics.get('f1_macro', 0),
                    'matthews_corrcoef': metrics.get('matthews_corrcoef', 0)
                })
        
        # Sort by accuracy
        performance_data.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return {
            'model_rankings': performance_data,
            'best_model': performance_data[0]['model'] if performance_data else None,
            'performance_spread': {
                'accuracy_range': [
                    min(m['accuracy'] for m in performance_data),
                    max(m['accuracy'] for m in performance_data)
                ] if performance_data else [0, 0],
                'top_3_models': [m['model'] for m in performance_data[:3]]
            }
        }
    
    def _generate_cv_summary(self, cv_results):
        """Generate cross-validation summary"""
        if not cv_results:
            return {}
        
        cv_summary = {}
        for model_name, results in cv_results.items():
            if isinstance(results, dict):
                cv_summary[model_name] = {
                    'mean_accuracy': results.get('accuracy_mean', 0),
                    'std_accuracy': results.get('accuracy_std', 0),
                    'mean_f1': results.get('f1_mean', 0),
                    'std_f1': results.get('f1_std', 0),
                    'stability_score': 1 - results.get('accuracy_std', 1)  # Lower std = higher stability
                }
        
        return cv_summary
    
    def _generate_tuning_summary(self, tuning_results):
        """Generate hyperparameter tuning summary"""
        if not tuning_results:
            return {}
        
        tuning_summary = {}
        for model_name, results in tuning_results.items():
            if isinstance(results, dict):
                tuning_summary[model_name] = {
                    'best_cv_score': results.get('best_score', 0),
                    'best_parameters': results.get('best_params', {}),
                    'improvement_potential': results.get('best_score', 0) - 0.5  # Baseline comparison
                }
        
        return tuning_summary
    
    def _generate_best_model_summary(self, best_model_info):
        """Generate best model summary"""
        if not best_model_info:
            return {}
        
        return {
            'model_name': best_model_info.get('name', 'Unknown'),
            'final_accuracy': best_model_info.get('accuracy', 0),
            'key_strengths': self._identify_model_strengths(best_model_info),
            'recommended_use_cases': self._get_use_case_recommendations(best_model_info.get('name', ''))
        }
    
    def _generate_recommendations(self, evaluation_results, validation_results):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        if validation_results:
            quality_score = validation_results.get('quality_score', 100)
            if quality_score < 80:
                recommendations.append({
                    'category': 'Data Quality',
                    'priority': 'High',
                    'recommendation': 'Improve data quality by addressing missing values, outliers, and class imbalance',
                    'impact': 'Could improve model performance by 5-15%'
                })
        
        # Model performance recommendations
        if evaluation_results:
            accuracies = [r.get('accuracy', 0) for r in evaluation_results.values() if isinstance(r, dict)]
            if accuracies and max(accuracies) < 0.85:
                recommendations.append({
                    'category': 'Model Performance',
                    'priority': 'Medium',
                    'recommendation': 'Consider ensemble methods or advanced algorithms (XGBoost, Neural Networks)',
                    'impact': 'Potential accuracy improvement of 3-10%'
                })
        
        # Feature engineering recommendations
        recommendations.append({
            'category': 'Feature Engineering',
            'priority': 'Medium',
            'recommendation': 'Explore feature interactions and polynomial features',
            'impact': 'May capture non-linear relationships and improve performance'
        })
        
        return recommendations
    
    def _generate_execution_summary(self, execution_times):
        """Generate execution time summary"""
        if not execution_times:
            return {}
        
        return {
            'total_runtime': execution_times.get('total', 0),
            'training_time': execution_times.get('training', 0),
            'tuning_time': execution_times.get('tuning', 0),
            'evaluation_time': execution_times.get('evaluation', 0),
            'performance_bottlenecks': self._identify_bottlenecks(execution_times)
        }
    
    def _identify_model_strengths(self, model_info):
        """Identify key strengths of the best model"""
        model_name = model_info.get('name', '').lower()
        
        strengths_map = {
            'random forest': ['Handles mixed data types well', 'Provides feature importance', 'Robust to outliers'],
            'gradient boosting': ['High predictive accuracy', 'Handles complex patterns', 'Good with structured data'],
            'svm': ['Effective in high dimensions', 'Memory efficient', 'Versatile with kernels'],
            'logistic regression': ['Fast and interpretable', 'No hyperparameter tuning needed', 'Probabilistic output'],
            'knn': ['Simple and intuitive', 'No assumptions about data', 'Works well with local patterns']
        }
        
        for key, strengths in strengths_map.items():
            if key in model_name:
                return strengths
        
        return ['Model-specific strengths not identified']
    
    def _get_use_case_recommendations(self, model_name):
        """Get use case recommendations for the model"""
        model_name = model_name.lower()
        
        use_cases_map = {
            'random forest': ['Feature selection studies', 'Baseline model development', 'Interpretable predictions'],
            'gradient boosting': ['Competition/production models', 'Complex pattern recognition', 'High-stakes predictions'],
            'svm': ['Text classification', 'High-dimensional data', 'Small to medium datasets'],
            'logistic regression': ['Quick prototypes', 'Baseline comparisons', 'Probability estimation'],
            'knn': ['Recommendation systems', 'Anomaly detection', 'Local pattern analysis']
        }
        
        for key, use_cases in use_cases_map.items():
            if key in model_name:
                return use_cases
        
        return ['General classification tasks']
    
    def _identify_bottlenecks(self, execution_times):
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        total_time = execution_times.get('total', 1)
        tuning_time = execution_times.get('tuning', 0)
        
        if tuning_time / total_time > 0.5:
            bottlenecks.append('Hyperparameter tuning takes >50% of total time')
        
        if total_time > 300:  # 5 minutes
            bottlenecks.append('Overall execution time is high - consider data sampling or model selection')
        
        return bottlenecks if bottlenecks else ['No significant bottlenecks identified']
    
    def save_report(self, output_dir='results', formats=['json', 'html', 'txt']):
        """Save report in multiple formats"""
        if not self.report_data:
            raise ValueError("No report data available. Generate report first.")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = self.timestamp.strftime('%Y%m%d_%H%M%S')
        
        saved_files = []
        
        # JSON format
        if 'json' in formats:
            json_path = os.path.join(output_dir, f'ml_report_{timestamp_str}.json')
            with open(json_path, 'w') as f:
                json.dump(self.report_data, f, indent=2, default=str)
            saved_files.append(json_path)
        
        # HTML format
        if 'html' in formats:
            html_path = os.path.join(output_dir, f'ml_report_{timestamp_str}.html')
            self._save_html_report(html_path)
            saved_files.append(html_path)
        
        # Text format
        if 'txt' in formats:
            txt_path = os.path.join(output_dir, f'ml_report_{timestamp_str}.txt')
            self._save_text_report(txt_path)
            saved_files.append(txt_path)
        
        return saved_files
    
    def _save_html_report(self, filepath):
        """Save report as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎓 Academic Performance Prediction - ML Pipeline Report</h1>
                <p>Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Data Summary</h2>
                {self._format_data_summary_html()}
            </div>
            
            <div class="section">
                <h2>🏆 Model Performance</h2>
                {self._format_model_performance_html()}
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                {self._format_recommendations_html()}
            </div>
            
            <div class="section">
                <h2>⏱️ Execution Summary</h2>
                {self._format_execution_summary_html()}
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def _save_text_report(self, filepath):
        """Save report as plain text"""
        with open(filepath, 'w') as f:
            f.write("🎓 ACADEMIC PERFORMANCE PREDICTION - ML PIPELINE REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data Summary
            f.write("📊 DATA SUMMARY\n")
            f.write("-" * 20 + "\n")
            data_summary = self.report_data.get('data_summary', {})
            f.write(f"Dataset Shape: {data_summary.get('dataset_shape', 'Unknown')}\n")
            f.write(f"Data Quality Score: {data_summary.get('data_quality_score', 0):.1f}/100\n")
            f.write(f"Missing Values: {data_summary.get('missing_values', 0)}\n\n")
            
            # Model Performance
            f.write("🏆 MODEL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            performance = self.report_data.get('model_performance', {})
            rankings = performance.get('model_rankings', [])
            for i, model in enumerate(rankings[:5], 1):
                f.write(f"{i}. {model['model']}: {model['accuracy']:.4f}\n")
            f.write("\n")
            
            # Recommendations
            f.write("💡 RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            recommendations = self.report_data.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. [{rec['priority']}] {rec['recommendation']}\n")
                f.write(f"   Impact: {rec['impact']}\n\n")
    
    def _format_data_summary_html(self):
        """Format data summary for HTML"""
        data_summary = self.report_data.get('data_summary', {})
        return f"""
        <div class="metric">Dataset Shape: {data_summary.get('dataset_shape', 'Unknown')}</div>
        <div class="metric">Data Quality Score: {data_summary.get('data_quality_score', 0):.1f}/100</div>
        <div class="metric">Missing Values: {data_summary.get('missing_values', 0)}</div>
        <div class="metric">Duplicate Rows: {data_summary.get('duplicate_rows', 0)}</div>
        """
    
    def _format_model_performance_html(self):
        """Format model performance for HTML"""
        performance = self.report_data.get('model_performance', {})
        rankings = performance.get('model_rankings', [])
        
        if not rankings:
            return "<p>No model performance data available.</p>"
        
        html = "<table><tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>F1-Score</th></tr>"
        for i, model in enumerate(rankings, 1):
            html += f"<tr><td>{i}</td><td>{model['model']}</td><td>{model['accuracy']:.4f}</td><td>{model['f1_macro']:.4f}</td></tr>"
        html += "</table>"
        
        return html
    
    def _format_recommendations_html(self):
        """Format recommendations for HTML"""
        recommendations = self.report_data.get('recommendations', [])
        
        if not recommendations:
            return "<p>No recommendations available.</p>"
        
        html = ""
        for rec in recommendations:
            priority_color = {'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'}.get(rec['priority'], '#6c757d')
            html += f"""
            <div class="recommendation">
                <strong style="color: {priority_color};">[{rec['priority']}]</strong> {rec['recommendation']}<br>
                <em>Impact: {rec['impact']}</em>
            </div>
            """
        
        return html
    
    def _format_execution_summary_html(self):
        """Format execution summary for HTML"""
        exec_summary = self.report_data.get('execution_summary', {})
        return f"""
        <div class="metric">Total Runtime: {exec_summary.get('total_runtime', 0):.2f} seconds</div>
        <div class="metric">Training Time: {exec_summary.get('training_time', 0):.2f} seconds</div>
        <div class="metric">Tuning Time: {exec_summary.get('tuning_time', 0):.2f} seconds</div>
        """

import sys  # Add this import at the top