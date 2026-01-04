"""Advanced data preprocessing utilities"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class AdvancedPreprocessor:
    """Advanced data preprocessing utilities"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scalers = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values in dataset
        
        Args:
            df: Input dataframe
            strategy: Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            Dataframe with missing values handled
        """
        df_copy = df.copy()
        
        if strategy == 'mean':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        elif strategy == 'drop':
            df_copy = df_copy.dropna()
        
        return df_copy
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from dataset
        
        Args:
            df: Input dataframe
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dataframe with outliers removed
        """
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_copy[numeric_cols]))
            df_copy = df_copy[(z_scores < threshold).all(axis=1)]
        
        return df_copy
    
    def scale_features(self, X: np.ndarray, scaler_type: str = 'standard', 
                      fit: bool = True) -> np.ndarray:
        """Scale features using specified scaler
        
        Args:
            X: Feature matrix
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler
            
        Returns:
            Scaled feature matrix
        """
        if scaler_type not in self.scalers:
            if scaler_type == 'standard':
                self.scalers[scaler_type] = StandardScaler()
            elif scaler_type == 'minmax':
                self.scalers[scaler_type] = MinMaxScaler()
            elif scaler_type == 'robust':
                self.scalers[scaler_type] = RobustScaler()
        
        scaler = self.scalers[scaler_type]
        
        if fit:
            return scaler.fit_transform(X)
        else:
            return scaler.transform(X)
    
    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistical summary of features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with feature statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats_dict = {
            'Feature': numeric_cols,
            'Mean': [df[col].mean() for col in numeric_cols],
            'Std': [df[col].std() for col in numeric_cols],
            'Min': [df[col].min() for col in numeric_cols],
            'Max': [df[col].max() for col in numeric_cols],
            'Median': [df[col].median() for col in numeric_cols],
            'Skewness': [df[col].skew() for col in numeric_cols],
            'Kurtosis': [df[col].kurtosis() for col in numeric_cols]
        }
        
        return pd.DataFrame(stats_dict).round(4)
    
    def detect_data_quality_issues(self, df: pd.DataFrame) -> dict:
        """Detect data quality issues
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with detected issues
        """
        issues = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_columns': len(df.columns) - len(df.columns.unique()),
            'constant_columns': [col for col in df.columns if df[col].nunique() == 1],
            'high_cardinality_columns': [col for col in df.select_dtypes(include=['object']).columns 
                                        if df[col].nunique() > df.shape[0] * 0.5]
        }
        
        return issues
    
    def print_data_quality_report(self, df: pd.DataFrame) -> None:
        """Print data quality report
        
        Args:
            df: Input dataframe
        """
        issues = self.detect_data_quality_issues(df)
        
        print("\n📊 Data Quality Report:")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print(f"Duplicate rows: {issues['duplicate_rows']}")
        print(f"Duplicate columns: {issues['duplicate_columns']}")
        
        if issues['missing_values']:
            print("\nMissing values:")
            for col, count in issues['missing_values'].items():
                if count > 0:
                    print(f"  {col}: {count}")
        
        if issues['constant_columns']:
            print(f"\nConstant columns: {issues['constant_columns']}")
        
        if issues['high_cardinality_columns']:
            print(f"\nHigh cardinality columns: {issues['high_cardinality_columns']}")
