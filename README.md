# Game of Thrones Character Prediction

A machine learning project for predicting Game of Thrones character survival using the *A Song of Ice and Fire* dataset. This project demonstrates data preprocessing, feature engineering, and model training with DecisionTreeClassifier.

## Overview

This project uses the Game of Thrones character dataset from the [Kaggle Topic Test Competition](https://www.kaggle.com/competitions/topic-test/leaderboard) to build a predictive model that determines whether a character died in the series based on features like house allegiance, gender, nobility, book appearances, and more. The model is trained using DecisionTreeClassifier and optimized with hyperparameter tuning using RandomizedSearchCV.

## Dataset

The dataset is sourced from the [Kaggle Topic Test Competition](https://www.kaggle.com/competitions/topic-test/leaderboard), which uses character data from *A Song of Ice and Fire* series. The competition provides:

- `train.csv`: Training data with character survival labels
- `test.csv`: Test data without survival labels  
- `sample_submission.csv`: Sample submission file format

The dataset contains metadata about characters including house allegiance, gender, nobility, appearances across books, and death-related information. Each row represents a single character from the Game of Thrones universe.

### Key Features:
- **Name**: Character name
- **Allegiances**: Character house (e.g., "Stark", "Lannister", "Night's Watch")
- **Death Year**: Year character died (NaN if alive)
- **Book of Death**: Book in which character died (1-5 representing the five books)
- **Death Chapter**: Chapter in which the character died
- **Book Intro Chapter**: Chapter character was introduced
- **Gender**: 1 = Male, 0 = Female
- **Nobility**: 1 = Noble, 0 = Commoner
- **GoT, CoK, SoS, FfC, DwD**: Binary indicators for appearance in each of the five books

Please download the dataset from the [Kaggle Topic Test Competition](https://www.kaggle.com/competitions/topic-test/data) and place the CSV files in the project `data/` directory.

## Features

- Data loading and exploration of Game of Thrones character dataset
- Handling missing values with zero-filling and categorical encoding with one-hot encoding
- Feature selection and preprocessing for character attributes
- DecisionTreeClassifier model training with interpretable decision rules
- Hyperparameter tuning with RandomizedSearchCV (10-fold cross-validation)
- Model evaluation using AUC, precision, recall, and accuracy metrics
- Decision tree visualization using Graphviz
- Prediction generation and submission file creation

## Installation

### Prerequisites

- Python 3.10 or higher
- uv (for dependency management)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/whats2000/Game-of-Thrones-Prediction.git
   cd Game-of-Thrones-Prediction
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   uv run python --version
   ```

## Usage

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   uv run jupyter notebook
   ```

2. Open `prediction.ipynb` and run the cells sequentially.

### Key Steps in the Notebook

1. **Load Data**: Import and explore the Game of Thrones character dataset
2. **Preprocess Data**: Handle missing values with zero-filling, encode categorical variables (Allegiances) with one-hot encoding
3. **Train Model**: Train a DecisionTreeClassifier with max_depth=4 for interpretability
4. **Hyperparameter Tuning**: Optimize model parameters using RandomizedSearchCV with 10-fold cross-validation
5. **Evaluate Model**: Assess performance using AUC, confusion matrix, precision, recall, and accuracy
6. **Visualize Tree**: Generate decision tree visualization using Graphviz for model interpretation
7. **Make Predictions**: Generate predictions for the test set characters
8. **Create Submission**: Save predictions to `submission.csv` in the required format

### Direct Execution

You can also run the notebook cells directly using nbconvert:

```bash
uv run jupyter nbconvert --to notebook --execute prediction.ipynb
```

## Dependencies

- `ipykernel`: Jupyter kernel for Python
- `notebook`: Jupyter Notebook
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning library with DecisionTreeClassifier
- `tqdm`: Progress bars for data processing
- `matplotlib`: Plotting and visualization
- `graphviz`: Decision tree visualization

## Model Performance

The DecisionTreeClassifier model achieves the following performance metrics on character survival prediction:

### Initial Model (max_depth=4):
- **AUC Score**: 0.6778
- **Precision**: 0.5246
- **Recall**: 0.5333
- **Accuracy**: 0.6686

### After Hyperparameter Tuning:
- **AUC Score**: 0.6865 (improvement of +0.0087)
- **Precision**: 0.5082
- **Recall**: 0.5167  
- **Accuracy**: 0.6570
- **Best CV Score**: 0.6655

**Optimal Parameters Found:**
- `max_depth`: 3
- `min_samples_split`: 3
- `min_samples_leaf`: 5
- `criterion`: 'gini'
- `ccp_alpha`: 0.001
- `class_weight`: None

The hyperparameter tuning through RandomizedSearchCV with 10-fold cross-validation tested 40 different parameter combinations (400 total fits). While the AUC score improved, the model maintains good interpretability with a maximum depth of 3, allowing clear understanding of the decision rules for character survival prediction.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for hosting the Topic Test competition with Game of Thrones character data
- Myles O'Neill for creating the original [Game of Thrones Dataset](https://www.kaggle.com/datasets/mylesoneill/game-of-thrones)
- George R.R. Martin for creating the rich *A Song of Ice and Fire* universe
- The open-source community for the machine learning libraries used

## Kaggle Competition

This project is designed for the [Kaggle Topic Test Competition](https://www.kaggle.com/competitions/topic-test/leaderboard). The notebook can be adapted for direct use on Kaggle by updating the data paths and ensuring all dependencies are available in the Kaggle environment.
