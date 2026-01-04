from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Tuple, Any
import joblib
import os

class ModelTrainer:
    def __init__(self) -> None:
        self.models: Dict[str, Any] = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        self.trained_models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
    
    def evaluate_models(self, X_test, y_test, target_encoder):
        """Evaluate all trained models"""
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Convert back to original labels for report
            y_test_labels = target_encoder.inverse_transform(y_test)
            y_pred_labels = target_encoder.inverse_transform(y_pred)
            
            self.results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test_labels, y_pred_labels),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(self.results[name]['classification_report'])
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the model with highest accuracy"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        return best_model_name, self.trained_models[best_model_name]
    
    def save_model(self, model_name: str, model: Any, filepath: str) -> None:
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def get_performance_summary(self) -> str:
        """Get a comprehensive performance summary"""
        if not self.results:
            return "No model results available"
        
        summary = "\n🏆 MODEL PERFORMANCE SUMMARY\n"
        summary += "=" * 50 + "\n"
        
        # Sort models by accuracy
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['accuracy'], 
                             reverse=True)
        
        for i, (name, result) in enumerate(sorted_models, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            summary += f"{medal} {name}: {result['accuracy']:.4f}\n"
        
        # Performance insights
        best_acc = sorted_models[0][1]['accuracy']
        worst_acc = sorted_models[-1][1]['accuracy']
        avg_acc = sum(r['accuracy'] for _, r in self.results.items()) / len(self.results)
        
        summary += f"\n📊 Performance Insights:\n"
        summary += f"   Best: {best_acc:.4f} | Worst: {worst_acc:.4f} | Average: {avg_acc:.4f}\n"
        summary += f"   Performance Spread: {(best_acc - worst_acc):.4f}\n"
        
        if best_acc > 0.9:
            summary += "   🎯 Excellent performance achieved!\n"
        elif best_acc > 0.8:
            summary += "   ✅ Good performance achieved!\n"
        else:
            summary += "   ⚠️  Consider feature engineering or advanced models\n"
        
        return summary