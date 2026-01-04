# 🎓 Academic Performance Prediction

[![Project Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/ipshita6-hub/AcadML)
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive machine learning project that predicts student academic performance using various classification algorithms. This project demonstrates a complete ML pipeline from data generation to model deployment.

## 🚀 Features

- **Synthetic Data Generation**: Creates realistic academic performance datasets
- **Multiple ML Models**: Implements 5 different classification algorithms
- **Comprehensive Evaluation**: Accuracy, classification reports, and confusion matrices
- **Enhanced Visualizations**: Interactive plots with Plotly, ROC curves, and precision-recall analysis
- **Advanced Metrics**: ROC-AUC scores, precision-recall curves, and detailed confusion matrices
- **Model Persistence**: Saves the best performing model for future use
- **Interactive Exploration**: Jupyter notebook for data analysis

## 📊 Results

The project achieves the following performance:

| Model | Accuracy |
|-------|----------|
| **Logistic Regression** | **84.5%** |
| SVM | 83.0% |
| KNN | 82.5% |
| Random Forest | 82.0% |
| Gradient Boosting | 79.5% |

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/ipshita6-hub/AcadML.git
cd AcadML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Quick Start
Run the complete analysis:
```bash
python main.py
```

### Interactive Analysis
Launch Jupyter notebook for exploration:
```bash
jupyter notebook notebooks/exploration.ipynb
```

### Custom Data
To use your own dataset, modify the `DataLoader` class in `src/data_loader.py`:
```python
data_loader = DataLoader(data_path='your_dataset.csv')
```

## 📁 Project Structure

```
AcadML/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── models.py                # ML model implementations
│   └── visualizer.py            # Visualization utilities
├── notebooks/                   # Jupyter notebooks
│   └── exploration.ipynb        # Interactive data exploration
├── models/                      # Trained model files
│   └── best_model_*.pkl         # Saved best model
├── results/                     # Generated visualizations
│   ├── data_distribution.png
│   ├── model_comparison.png
│   └── confusion_matrices.png
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── LICENSE                      # MIT License
```

## 🔍 Dataset Features

The synthetic dataset includes the following features:

- **study_hours**: Average daily study hours
- **attendance_rate**: Class attendance percentage (0.6-1.0)
- **previous_grade**: Previous academic performance (0-100)
- **family_income**: Categorical (Low, Medium, High)
- **extracurricular**: Binary (0: No, 1: Yes)
- **parent_education**: Categorical (High School, Bachelor, Master, PhD)
- **performance**: Target variable (Poor, Average, Good)

## 🤖 Models Implemented

1. **Random Forest**: Ensemble method with decision trees
2. **Gradient Boosting**: Sequential ensemble learning
3. **Support Vector Machine (SVM)**: Kernel-based classification
4. **Logistic Regression**: Linear probabilistic classifier
5. **K-Nearest Neighbors (KNN)**: Instance-based learning

## 📈 Enhanced Visualizations

The project now includes both traditional and interactive visualizations:

### Interactive Visualizations (HTML)
- **Interactive Data Distribution**: Hover-enabled histograms and pie charts
- **Correlation Heatmap**: Interactive feature correlation matrix
- **Model Comparison**: Interactive bar charts with detailed metrics
- **Feature Importance**: Comparative analysis across tree-based models
- **Performance Dashboard**: Comprehensive multi-panel overview

### Advanced Analysis Plots
- **ROC Curves**: Receiver Operating Characteristic curves for all models
- **Precision-Recall Curves**: Detailed performance analysis
- **Enhanced Confusion Matrices**: With both counts and percentages
- **Feature Correlation**: Triangle matrix and full interactive heatmap

### Traditional Visualizations (PNG)
- **Data Distribution**: Histograms and pie charts of feature distributions
- **Model Comparison**: Bar chart comparing model accuracies
- **Confusion Matrices**: Heatmaps showing prediction vs actual results
- **Feature Importance**: Bar plots for tree-based models

## 🔧 Customization

### Adding New Models
Add new classifiers in `src/models.py`:
```python
self.models['New Model'] = YourClassifier()
```

### Custom Features
Modify the data generation in `src/data_loader.py`:
```python
def generate_sample_data(self, n_samples=1000):
    # Add your custom features here
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/)
- Visualizations powered by [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/)
- Interactive analysis with [Jupyter](https://jupyter.org/)

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ **Star this repository if you found it helpful!**
