from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        self.trained_models = {}
        self.results = {}
    
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
    
    def get_best_model(self):
        """Get the model with highest accuracy"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        return best_model_name, self.trained_models[best_model_name]
    
    def save_model(self, model_name, model, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load saved model"""
        return joblib.load(filepath)