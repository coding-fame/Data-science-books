# Train & Test Datasets in Python

---

## Types of Datasets in Machine Learning
Machine learning models use three primary types of datasets:

- **Train Dataset**
- **Test Dataset**
- **Validation Dataset** (optional)

> **Note:** While the validation dataset is optional, training and test datasets are mandatory.

## What Are Train and Test Datasets?

In machine learning, datasets are typically divided into two main subsets:

- **Training Dataset**: Used to train the model, allowing it to learn patterns and relationships in the data.
- **Test Dataset**: Used to evaluate the model's performance on unseen data, assessing how well it generalizes.
- **Validation Dataset (Optional)**: Used to fine-tune the model, often in hyperparameter tuning.

## How to Decide Dataset Sizes
There is no strict rule, but experts recommend:

| Train Set | Test Set |
|-----------|---------|
| **70%**   | **30%** |
| **60%**   | **40%** |

- The exact ratio depends on the size and nature of the dataset, but these proportions are often a good starting point.

## Splitting Data Using `train_test_split()`
The `train_test_split()` function from `sklearn.model_selection` is used to split datasets into training and testing sets.

---

## Basic Train-Test Split
```python
import numpy as np
from sklearn.model_selection import train_test_split

dataset = np.arange(10)
X_train, X_test = train_test_split(dataset, train_size=0.6)

print("Train Data:", X_train)
print("Test Data:", X_test)
```

---

## Tools and Libraries

We'll use the following Python libraries:
- **`scikit-learn`**: For dataset splitting, preprocessing, modeling, and cross-validation.
- **`pandas`**: For data manipulation and handling structured data.
- **`numpy`**: For numerical operations and creating synthetic datasets.

---

## 1. Loading a Dataset

Let's start by loading datasets to work with. We'll use the **Iris dataset** for classification examples and the **California Housing dataset** for regression examples, both available in `scikit-learn`.

```python
from sklearn.datasets import load_iris, fetch_california_housing
import pandas as pd
import numpy as np

# Load the Iris dataset (classification)
iris = load_iris()
X_iris = iris.data  # Features
y_iris = iris.target  # Labels

# Load the California Housing dataset (regression)
housing = fetch_california_housing()
X_housing = housing.data  # Features
y_housing = housing.target  # Target variable (house prices)
```

- **Iris Dataset**: Contains 150 samples with 4 features (sepal/petal length and width) and 3 classes (species).
- **California Housing Dataset**: Contains 20,640 samples with 8 features (e.g., house age, number of rooms) and a continuous target (house price).

---

## 2. Random Splitting
The simplest way to split data is **random splitting**, where data is divided randomly into training and test sets. A common ratio is 80% training and 20% testing. The `train_test_split` function from `scikit-learn` makes this easy.

### Ensuring Consistency with `random_state`
Using `random_state=0` ensures that the same train-test split is obtained across multiple runs.

💡 Use 42 as a convention, but any fixed number works! 🚀

```python
from sklearn.model_selection import train_test_split

# Split the Iris dataset (80% train, 20% test)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

print(f"Iris training set size: {X_train_iris.shape[0]} samples")
print(f"Iris test set size: {X_test_iris.shape[0]} samples")
```

- **Output** (approximate):
  ```
  Iris training set size: 120 samples
  Iris test set size: 30 samples
  ```
- **`test_size=0.2`**: 20% of the data goes to the test set.
- **`random_state=42`**: Ensures reproducibility.

---

## 3. Stratified Splitting

For classification problems, especially with imbalanced datasets, **stratified splitting** ensures that the class distribution is preserved in both training and test sets. The `train_test_split` function supports this via the `stratify` parameter.

```python
# Stratified split for Iris dataset
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X_iris, y_iris, test_size=0.2, stratify=y_iris, random_state=42
)

# Check class distribution
print("Original class distribution:")
print(pd.Series(y_iris).value_counts(normalize=True))
print("Training set class distribution:")
print(pd.Series(y_train_strat).value_counts(normalize=True))
print("Test set class distribution:")
print(pd.Series(y_test_strat).value_counts(normalize=True))
```

- **Output** (approximate):
  ```
  Original class distribution:
  0    0.333333
  1    0.333333
  2    0.333333
  Name: proportion, dtype: float64
  Training set class distribution:
  0    0.333333
  1    0.333333
  2    0.333333
  Name: proportion, dtype: float64
  Test set class distribution:
  0    0.333333
  1    0.333333
  2    0.333333
  Name: proportion, dtype: float64
  ```
- The proportions remain consistent across splits, which is critical for fair evaluation.

---

## 4. Time-Based Splitting for Time Series Data

For **time series data**, random splitting is inappropriate because the temporal order matters. Instead, we split based on time, using earlier data for training and later data for testing.

Here’s an example with a synthetic time series dataset:

```python
# Create a synthetic time series dataset
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'target': np.random.randn(100)
}, index=dates)

# Split: first 80 days for training, last 20 for testing
train_size = 80
X_time = data[['feature1', 'feature2']]
y_time = data['target']
X_train_time = X_time.iloc[:train_size]
X_test_time = X_time.iloc[train_size:]
y_train_time = y_time.iloc[:train_size]
y_test_time = y_time.iloc[train_size:]

print(f"Time-based training set size: {X_train_time.shape[0]} samples")
print(f"Time-based test set size: {X_test_time.shape[0]} samples")
```

- **Output**:
  ```
  Time-based training set size: 80 samples
  Time-based test set size: 20 samples
  ```
- The split preserves the temporal sequence, mimicking real-world forecasting scenarios.

---

## 5. Cross-Validation

**Cross-validation** provides a robust estimate of model performance by splitting the data into multiple "folds" and training/testing the model multiple times. Common methods include `KFold` (for regression) and `StratifiedKFold` (for classification).

**K-Fold Cross Validation** solves the limitations of a single train-test split by splitting the dataset into **K** equal-sized folds. The model is trained and evaluated **K** times, ensuring every data point is used for both training and validation.

### How K-Fold Works
1. Divide the dataset into **K** equal-sized subsets (folds).
2. Use **one** fold as the validation set and the remaining **K-1** folds for training.
3. Train and evaluate the model.
4. Repeat the process **K** times, each time using a different fold as the validation set.
5. Compute the average of all K iterations to get a final performance score.

### Advantages of K-Fold Cross Validation
- Reduces **bias** and **variance**.
- Ensures **every data point** is used for both training and validation.
- Provides a **more accurate estimate** of model performance.
- Avoids **overfitting** compared to a simple train-test split.

### 5.1. Stratified K-Fold for Classification

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Define the model
model = RandomForestClassifier(random_state=42)

# Define StratifiedKFold (5 folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model using K-Fold
accuracies = cross_val_score(model, X_iris, y_iris, cv=skf)

print(f"Cross-validation accuracies: {accuracies}")
print(f"Mean accuracy: {np.mean(accuracies):.2f}")
```

- **Output** (example):
  ```
  Cross-validation accuracies: [0.9667, 0.9333, 0.9667, 0.9667, 1.0]
  Mean accuracy: 0.97
  ```
- Each fold uses a different test set, and the mean accuracy reflects overall performance.

### 5.2. K-Fold for Regression

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the model
model = LinearRegression()

# Define KFold (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate model using K-Fold
scores = cross_val_score(model, X_housing, y_housing, cv=kf)

# Perform cross-validation on California Housing dataset

print(f"Cross-validation MSEs: {scores}")
print(f"Mean MSE: {np.mean(scores):.2f}")
```

- **Output** (example):
  ```
  Accuracy with pipeline: 0.97
  ```
- The pipeline ensures that scaling is applied consistently, avoiding manual errors.

---

## Summary of Methods and Tools

| **Method**            | **Tool/Function**           | **Use Case**                          | **Key Feature**                     |
|-----------------------|-----------------------------|---------------------------------------|-------------------------------------|
| Random Split          | `train_test_split`          | General-purpose splitting            | Simple, fast                        |
| Stratified Split      | `train_test_split(stratify)`| Classification with imbalanced data  | Preserves class distribution        |
| Time-Based Split      | Manual (e.g., pandas)       | Time series data                     | Maintains temporal order            |
| Cross-Validation      | `KFold`, `StratifiedKFold`  | Robust performance estimation        | Multiple train/test splits          |
| Preprocessing         | `StandardScaler`, `Pipeline`| Feature scaling, avoiding leakage    | Safe preprocessing application      |

---

## Conclusion

This guide has covered the essentials of handling train and test datasets in Python:
- **Loading datasets** using `scikit-learn`.
- **Splitting methods**: Random, stratified, time-based, and cross-validation.
- **Preprocessing**: Proper techniques to avoid data leakage using scalers and pipelines.
- **Examples**: Practical code for classification (Iris), regression (California Housing), and time series data.

---

## Cross-Validation and Pipelines

### Why Use a Pipeline?
- Ensures preprocessing occurs **after** each cross-validation split.
- Prevents **data leakage**.

### Cross-Validate a Pipeline
```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('http://bit.ly/kaggletrain')
X = df[['Sex', 'Name']]
y = df['Survived']

ohe = OneHotEncoder()
vect = CountVectorizer()
ct = make_column_transformer((ohe, ['Sex']), (vect, 'Name'))
clf = LogisticRegression(solver='liblinear', random_state=1)
pipe = make_pipeline(ct, clf)

cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
```
