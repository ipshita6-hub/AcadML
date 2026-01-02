import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic academic performance data"""
        np.random.seed(42)
        
        data = {
            'study_hours': np.random.normal(5, 2, n_samples),
            'attendance_rate': np.random.uniform(0.6, 1.0, n_samples),
            'previous_grade': np.random.normal(75, 15, n_samples),
            'family_income': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'extracurricular': np.random.choice([0, 1], n_samples),
            'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
        }
        
        # Create performance based on features
        performance_score = (
            data['study_hours'] * 2 +
            data['attendance_rate'] * 30 +
            data['previous_grade'] * 0.3 +
            np.random.normal(0, 5, n_samples)
        )
        
        # Convert to categories
        data['performance'] = ['Poor' if x < 60 else 'Average' if x < 80 else 'Good' 
                              for x in performance_score]
        
        return pd.DataFrame(data)
    
    def load_data(self):
        """Load or generate data"""
        if self.data_path:
            return pd.read_csv(self.data_path)
        else:
            return self.generate_sample_data()
    
    def preprocess_data(self, df):
        """Preprocess the data for ML"""
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop('performance')  # Don't encode target
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Separate features and target
        X = df.drop('performance', axis=1)
        y = df['performance']
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.target_encoder = target_encoder
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_encoded, X.columns.tolist()
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)