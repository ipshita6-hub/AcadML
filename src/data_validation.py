import pandas as pd
import numpy as np
from scipy import stats
import warnings

class DataValidator:
    def __init__(self):
        self.validation_results = {}
        self.data_quality_report = {}
        
    def validate_dataset(self, df, target_column='performance'):
        """Comprehensive dataset validation"""
        print("\n🔍 Performing data validation...")
        
        self.validation_results = {
            'basic_info': self._check_basic_info(df),
            'missing_values': self._check_missing_values(df),
            'duplicates': self._check_duplicates(df),
            'data_types': self._check_data_types(df),
            'outliers': self._detect_outliers(df),
            'class_balance': self._check_class_balance(df, target_column),
            'feature_correlations': self._check_feature_correlations(df),
            'data_distribution': self._check_data_distribution(df)
        }
        
        return self.validation_results
    
    def _check_basic_info(self, df):
        """Check basic dataset information"""
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'column_count': len(df.columns),
            'row_count': len(df)
        }
    
    def _check_missing_values(self, df):
        """Check for missing values"""
        missing_info = {}
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        for col in df.columns:
            missing_info[col] = {
                'count': missing_counts[col],
                'percentage': missing_percentages[col]
            }
        
        total_missing = missing_counts.sum()
        return {
            'total_missing': total_missing,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_by_column': missing_info,
            'has_missing': total_missing > 0
        }
    
    def _check_duplicates(self, df):
        """Check for duplicate rows"""
        duplicate_count = df.duplicated().sum()
        return {
            'duplicate_count': duplicate_count,
            'duplicate_percentage': (duplicate_count / len(df)) * 100,
            'has_duplicates': duplicate_count > 0
        }
    
    def _check_data_types(self, df):
        """Check data types and suggest improvements"""
        type_info = {}
        for col in df.columns:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            
            type_info[col] = {
                'current_type': str(dtype),
                'unique_values': unique_count,
                'sample_values': df[col].dropna().head(3).tolist()
            }
            
            # Suggest optimizations
            if dtype == 'object' and unique_count < 10:
                type_info[col]['suggestion'] = 'Consider converting to category'
            elif dtype == 'float64' and df[col].dropna().apply(lambda x: x.is_integer()).all():
                type_info[col]['suggestion'] = 'Consider converting to int'
        
        return type_info
    
    def _detect_outliers(self, df):
        """Detect outliers using IQR method"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outlier_info[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': outliers.index.tolist()
            }
        
        return outlier_info
    
    def _check_class_balance(self, df, target_column):
        """Check class balance in target variable"""
        if target_column not in df.columns:
            return {'error': f'Target column {target_column} not found'}
        
        class_counts = df[target_column].value_counts()
        class_percentages = df[target_column].value_counts(normalize=True) * 100
        
        # Calculate imbalance ratio
        max_class = class_counts.max()
        min_class = class_counts.min()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        return {
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_percentages.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'is_balanced': imbalance_ratio <= 2.0,  # Threshold for balanced dataset
            'minority_class': class_counts.idxmin(),
            'majority_class': class_counts.idxmax()
        }
    
    def _check_feature_correlations(self, df):
        """Check for highly correlated features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {'error': 'Not enough numerical columns for correlation analysis'}
        
        corr_matrix = df[numerical_cols].corr()
        
        # Find highly correlated pairs (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr_pairs,
            'has_multicollinearity': len(high_corr_pairs) > 0
        }
    
    def _check_data_distribution(self, df):
        """Check data distribution normality"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        distribution_info = {}
        
        for col in numerical_cols:
            # Shapiro-Wilk test for normality (for small samples)
            if len(df[col].dropna()) <= 5000:
                try:
                    stat, p_value = stats.shapiro(df[col].dropna())
                    is_normal = p_value > 0.05
                except:
                    stat, p_value, is_normal = None, None, None
            else:
                # Use Kolmogorov-Smirnov test for larger samples
                try:
                    stat, p_value = stats.kstest(df[col].dropna(), 'norm')
                    is_normal = p_value > 0.05
                except:
                    stat, p_value, is_normal = None, None, None
            
            distribution_info[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'is_normal': is_normal,
                'normality_p_value': p_value
            }
        
        return distribution_info
    
    def print_validation_report(self):
        """Print comprehensive validation report"""
        if not self.validation_results:
            print("No validation results available. Run validate_dataset() first.")
            return
        
        print("\n📋 Data Validation Report")
        print("=" * 80)
        
        # Basic Info
        basic = self.validation_results['basic_info']
        print(f"\n📊 Dataset Overview:")
        print(f"  Shape: {basic['shape']}")
        print(f"  Memory Usage: {basic['memory_usage'] / 1024:.2f} KB")
        
        # Missing Values
        missing = self.validation_results['missing_values']
        print(f"\n❓ Missing Values:")
        if missing['has_missing']:
            print(f"  Total Missing: {missing['total_missing']}")
            for col, count in missing['columns_with_missing'].items():
                percentage = (count / basic['shape'][0]) * 100
                print(f"  {col}: {count} ({percentage:.2f}%)")
        else:
            print("  ✅ No missing values found")
        
        # Duplicates
        duplicates = self.validation_results['duplicates']
        print(f"\n🔄 Duplicate Rows:")
        if duplicates['has_duplicates']:
            print(f"  ⚠️  Found {duplicates['duplicate_count']} duplicates ({duplicates['duplicate_percentage']:.2f}%)")
        else:
            print("  ✅ No duplicate rows found")
        
        # Class Balance
        balance = self.validation_results['class_balance']
        print(f"\n⚖️  Class Balance:")
        if 'error' not in balance:
            print(f"  Imbalance Ratio: {balance['imbalance_ratio']:.2f}")
            print(f"  Balanced: {'✅ Yes' if balance['is_balanced'] else '⚠️  No'}")
            print(f"  Class Distribution:")
            for class_name, count in balance['class_counts'].items():
                percentage = balance['class_percentages'][class_name]
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        # Outliers
        outliers = self.validation_results['outliers']
        print(f"\n🎯 Outliers:")
        total_outliers = sum(info['outlier_count'] for info in outliers.values())
        if total_outliers > 0:
            print(f"  ⚠️  Found outliers in {len([col for col, info in outliers.items() if info['outlier_count'] > 0])} columns")
            for col, info in outliers.items():
                if info['outlier_count'] > 0:
                    print(f"    {col}: {info['outlier_count']} outliers ({info['outlier_percentage']:.2f}%)")
        else:
            print("  ✅ No outliers detected")
        
        # High Correlations
        correlations = self.validation_results['feature_correlations']
        print(f"\n🔗 Feature Correlations:")
        if 'error' not in correlations and correlations['has_multicollinearity']:
            print(f"  ⚠️  Found {len(correlations['high_correlations'])} highly correlated pairs")
            for pair in correlations['high_correlations']:
                print(f"    {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print("  ✅ No high correlations detected")
    
    def get_data_quality_score(self):
        """Calculate overall data quality score"""
        if not self.validation_results:
            return None
        
        score = 100  # Start with perfect score
        
        # Deduct for missing values
        missing = self.validation_results['missing_values']
        if missing['has_missing']:
            missing_penalty = min(missing['total_missing'] / (self.validation_results['basic_info']['shape'][0] * self.validation_results['basic_info']['shape'][1]) * 100, 20)
            score -= missing_penalty
        
        # Deduct for duplicates
        duplicates = self.validation_results['duplicates']
        if duplicates['has_duplicates']:
            duplicate_penalty = min(duplicates['duplicate_percentage'], 10)
            score -= duplicate_penalty
        
        # Deduct for class imbalance
        balance = self.validation_results['class_balance']
        if 'error' not in balance and not balance['is_balanced']:
            imbalance_penalty = min((balance['imbalance_ratio'] - 2) * 2, 15)
            score -= imbalance_penalty
        
        # Deduct for outliers
        outliers = self.validation_results['outliers']
        total_outlier_percentage = sum(info['outlier_percentage'] for info in outliers.values()) / len(outliers)
        outlier_penalty = min(total_outlier_percentage / 2, 10)
        score -= outlier_penalty
        
    def get_quality_insights(self):
        """Get actionable data quality insights"""
        if not self.validation_results:
            return "No validation results available"
        
        insights = []
        score = self.get_data_quality_score()
        
        # Overall assessment
        if score >= 90:
            insights.append("🌟 Excellent data quality! Your dataset is production-ready.")
        elif score >= 80:
            insights.append("✅ Good data quality with minor issues to address.")
        elif score >= 70:
            insights.append("⚠️  Moderate data quality. Consider data cleaning.")
        else:
            insights.append("❌ Poor data quality. Significant cleaning required.")
        
        # Specific recommendations
        missing = self.validation_results.get('missing_values', {})
        if missing.get('has_missing', False):
            insights.append(f"📝 Action: Handle {missing.get('total_missing', 0)} missing values")
        
        duplicates = self.validation_results.get('duplicates', {})
        if duplicates.get('has_duplicates', False):
            insights.append(f"🔄 Action: Remove {duplicates.get('duplicate_count', 0)} duplicate rows")
        
        balance = self.validation_results.get('class_balance', {})
        if not balance.get('is_balanced', True):
            ratio = balance.get('imbalance_ratio', 1)
            insights.append(f"⚖️  Action: Address class imbalance (ratio: {ratio:.1f}:1)")
        
        correlations = self.validation_results.get('feature_correlations', {})
        if correlations.get('has_multicollinearity', False):
            count = len(correlations.get('high_correlations', []))
            insights.append(f"🔗 Action: Review {count} highly correlated feature pairs")
        
        return "\n".join(insights)