import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import os

class EnhancedVisualizer:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8')
        
    def plot_interactive_data_distribution(self, df, save=True):
        """Create interactive data distribution plots using Plotly"""
        # Numerical features distribution
        numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('performance', errors='ignore')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'Distribution of {col}' for col in numerical_cols[:4]],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, col in enumerate(numerical_cols[:4]):
            row = i // 2 + 1
            col_idx = i % 2 + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, marker_color=colors[i], opacity=0.7),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title_text="Interactive Data Distribution Analysis",
            showlegend=False,
            height=600
        )
        
        if save:
            fig.write_html(f'{self.results_dir}/interactive_data_distribution.html')
            fig.write_image(f'{self.results_dir}/interactive_data_distribution.png')
        
        fig.show()
        
        # Performance distribution pie chart
        performance_counts = df['performance'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=performance_counts.index,
            values=performance_counts.values,
            hole=0.3,
            marker_colors=['#ff9999', '#66b3ff', '#99ff99']
        )])
        
        fig_pie.update_layout(
            title="Performance Distribution",
            annotations=[dict(text='Performance', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        if save:
            fig_pie.write_html(f'{self.results_dir}/performance_distribution.html')
            fig_pie.write_image(f'{self.results_dir}/performance_distribution.png')
        
        fig_pie.show()
    
    def plot_correlation_heatmap(self, df, save=True):
        """Create an enhanced correlation heatmap"""
        # Prepare data for correlation
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        # Simple label encoding for correlation analysis
        for col in categorical_cols:
            if col != 'performance':
                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        
        # Performance encoding
        performance_map = {'Poor': 0, 'Average': 1, 'Good': 2}
        df_encoded['performance'] = df_encoded['performance'].map(performance_map)
        
        # Calculate correlation matrix
        corr_matrix = df_encoded.corr()
        
        # Interactive heatmap with Plotly
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            width=800,
            height=600
        )
        
        if save:
            fig.write_html(f'{self.results_dir}/correlation_heatmap.html')
            fig.write_image(f'{self.results_dir}/correlation_heatmap.png')
        
        fig.show()
        
        # Traditional matplotlib version with better styling
        plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix (Lower Triangle)', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.results_dir}/correlation_matrix_triangle.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_model_comparison(self, results, save=True):
        """Create interactive model comparison with detailed metrics"""
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        # Interactive bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=accuracies,
                text=[f'{acc:.3f}' for acc in accuracies],
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names)],
                hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Model Accuracy Comparison',
            xaxis_title='Models',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1]),
            height=500
        )
        
        if save:
            fig.write_html(f'{self.results_dir}/interactive_model_comparison.html')
            fig.write_image(f'{self.results_dir}/interactive_model_comparison.png')
        
        fig.show()
    
    def plot_roc_curves(self, models, X_test, y_test, target_encoder, save=True):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(12, 8))
        
        # Get unique classes
        classes = target_encoder.classes_
        n_classes = len(classes)
        
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (name, model) in enumerate(models.items()):
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)
            elif hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
                # Normalize decision function output
                y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
            else:
                continue
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            if n_classes == 2:
                fpr[0], tpr[0], _ = roc_curve(y_test_bin, y_score[:, 1])
                roc_auc[0] = auc(fpr[0], tpr[0])
                plt.plot(fpr[0], tpr[0], color=colors[i % len(colors)], lw=2,
                        label=f'{name} (AUC = {roc_auc[0]:.3f})')
            else:
                for j in range(n_classes):
                    fpr[j], tpr[j], _ = roc_curve(y_test_bin[:, j], y_score[:, j])
                    roc_auc[j] = auc(fpr[j], tpr[j])
                
                # Compute micro-average ROC curve
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                plt.plot(fpr["micro"], tpr["micro"], color=colors[i % len(colors)], lw=2,
                        label=f'{name} (Micro-avg AUC = {roc_auc["micro"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(f'{self.results_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, models, X_test, y_test, target_encoder, save=True):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(12, 8))
        
        classes = target_encoder.classes_
        n_classes = len(classes)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (name, model) in enumerate(models.items()):
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)
            elif hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
                y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
            else:
                continue
            
            if n_classes == 2:
                precision, recall, _ = precision_recall_curve(y_test_bin, y_score[:, 1])
                avg_precision = auc(recall, precision)
                plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                        label=f'{name} (AP = {avg_precision:.3f})')
            else:
                # Compute micro-average precision-recall curve
                precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
                avg_precision = auc(recall, precision)
                plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                        label=f'{name} (Micro-avg AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=16)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(f'{self.results_dir}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_enhanced_confusion_matrices(self, results, target_encoder, save=True):
        """Create enhanced confusion matrices with better styling"""
        n_models = len(results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, result) in enumerate(results.items()):
            row, col = i // cols, i % cols
            cm = result['confusion_matrix']
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create annotations with both counts and percentages
            annotations = []
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    annotations.append(f'{cm[r,c]}\n({cm_percent[r,c]:.1f}%)')
            
            annotations = np.array(annotations).reshape(cm.shape)
            
            sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                       xticklabels=target_encoder.classes_,
                       yticklabels=target_encoder.classes_,
                       ax=axes[row, col], cbar_kws={'shrink': 0.8})
            
            axes[row, col].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}', fontsize=14)
            axes[row, col].set_xlabel('Predicted', fontsize=12)
            axes[row, col].set_ylabel('Actual', fontsize=12)
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Enhanced Confusion Matrices (Count & Percentage)', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.results_dir}/enhanced_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance_comparison(self, models, feature_names, save=True):
        """Compare feature importance across tree-based models"""
        importance_data = {}
        
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = model.feature_importances_
        
        if not importance_data:
            print("No tree-based models found for feature importance comparison")
            return
        
        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame(importance_data, index=feature_names)
        
        # Interactive plot with Plotly
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, model_name in enumerate(importance_df.columns):
            fig.add_trace(go.Bar(
                name=model_name,
                x=feature_names,
                y=importance_df[model_name],
                marker_color=colors[i % len(colors)],
                opacity=0.8
            ))
        
        fig.update_layout(
            title='Feature Importance Comparison Across Models',
            xaxis_title='Features',
            yaxis_title='Importance',
            barmode='group',
            height=600
        )
        
        if save:
            fig.write_html(f'{self.results_dir}/feature_importance_comparison.html')
            fig.write_image(f'{self.results_dir}/feature_importance_comparison.png')
        
        fig.show()
        
        # Traditional matplotlib version
        plt.figure(figsize=(14, 8))
        importance_df.plot(kind='bar', width=0.8)
        plt.title('Feature Importance Comparison Across Models', fontsize=16)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.results_dir}/feature_importance_comparison_static.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_performance_dashboard(self, results, save=True):
        """Create a comprehensive performance dashboard"""
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        # Create subplots with supported types
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy', 'Performance Radar', 'Model Ranking', 'Accuracy Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatterpolar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # 1. Model Accuracy Bar Chart
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name='Accuracy',
                  marker_color='lightblue', showlegend=False),
            row=1, col=1
        )
        
        # 2. Radar Chart (using scatterpolar)
        fig.add_trace(
            go.Scatterpolar(
                r=accuracies + [accuracies[0]],  # Close the polygon
                theta=model_names + [model_names[0]],
                fill='toself',
                name='Performance',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Model Ranking
        sorted_models = sorted(zip(model_names, accuracies), key=lambda x: x[1], reverse=True)
        ranks = list(range(1, len(sorted_models) + 1))
        sorted_names = [x[0] for x in sorted_models]
        sorted_accs = [x[1] for x in sorted_models]
        
        fig.add_trace(
            go.Bar(x=ranks, y=sorted_accs, text=sorted_names,
                  textposition='auto', name='Ranking', showlegend=False,
                  marker_color='lightgreen'),
            row=2, col=1
        )
        
        # 4. Accuracy Distribution
        fig.add_trace(
            go.Histogram(x=accuracies, nbinsx=10, name='Distribution',
                        showlegend=False, marker_color='lightcoral'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Model Performance Dashboard",
            height=800,
            showlegend=False
        )
        
        # Update x-axis labels for ranking
        fig.update_xaxes(title_text="Rank", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        
        if save:
            fig.write_html(f'{self.results_dir}/performance_dashboard.html')
            fig.write_image(f'{self.results_dir}/performance_dashboard.png')
        
        fig.show()