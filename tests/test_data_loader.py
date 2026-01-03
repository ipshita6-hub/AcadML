import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.data_loader = DataLoader()
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        df = self.data_loader.generate_sample_data(n_samples=100)
        
        # Check shape
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 7)
        
        # Check columns
        expected_columns = ['study_hours', 'attendance_rate', 'previous_grade', 
                          'family_income', 'extracurricular', 'parent_education', 'performance']
        self.assertEqual(list(df.columns), expected_columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df['study_hours']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['attendance_rate']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['previous_grade']))
        self.assertTrue(pd.api.types.is_object_dtype(df['family_income']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['extracurricular']))
        self.assertTrue(pd.api.types.is_object_dtype(df['parent_education']))
        self.assertTrue(pd.api.types.is_object_dtype(df['performance']))
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        df = self.data_loader.generate_sample_data(n_samples=50)
        X, y, feature_names = self.data_loader.preprocess_data(df)
        
        # Check shapes
        self.assertEqual(X.shape[0], 50)
        self.assertEqual(len(y), 50)
        self.assertEqual(len(feature_names), X.shape[1])
        
        # Check that X is numeric
        self.assertTrue(np.issubdtype(X.dtype, np.number))
        
        # Check that y is encoded
        self.assertTrue(np.issubdtype(y.dtype, np.integer))
        
        # Check that target encoder is created
        self.assertIsNotNone(self.data_loader.target_encoder)
    
    def test_split_data(self):
        """Test data splitting"""
        df = self.data_loader.generate_sample_data(n_samples=100)
        X, y, _ = self.data_loader.preprocess_data(df)
        
        X_train, X_test, y_train, y_test = self.data_loader.split_data(X, y, test_size=0.3)
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 70)
        self.assertEqual(X_test.shape[0], 30)
        self.assertEqual(len(y_train), 70)
        self.assertEqual(len(y_test), 30)
        
        # Check that features match
        self.assertEqual(X_train.shape[1], X_test.shape[1])
    
    def test_data_consistency(self):
        """Test data consistency across multiple generations"""
        # Generate data with same random state
        df1 = self.data_loader.generate_sample_data(n_samples=50)
        
        # Reset and generate again
        data_loader2 = DataLoader()
        df2 = data_loader2.generate_sample_data(n_samples=50)
        
        # Should be identical due to fixed random state
        pd.testing.assert_frame_equal(df1, df2)

if __name__ == '__main__':
    unittest.main()