
# K-Nearest Neighbors (K-NN) Algorithm

The **K-Nearest Neighbors (K-NN)** algorithm is a simple yet powerful supervised machine learning technique used for both **classification** and **regression** tasks. It’s an **instance-based learning** algorithm, meaning it doesn’t build a model during training but instead memorizes the training data and makes predictions by comparing new data points to stored examples. 

## **What is the K-Nearest Neighbors (K-NN) Algorithm?**
K-NN is a **non-parametric**, instance-based learning algorithm that classifies or predicts the value of a data point based on the majority class (for classification) or average value (for regression) of its \(k\) nearest neighbors in the feature space. 
It’s “lazy” because it doesn’t build a model during training—instead, it memorizes the training data and performs computations at prediction time.

*Instance based learning*
Instead of learning a fixed model (like a linear equation), it relies on the training data itself to make predictions. When a new data point is introduced, K-NN identifies the **K closest data points** (neighbors) from the training set and uses their labels or values to predict the outcome.

## **Why Use K-NN?**
- **Simplicity**: Intuitive and easy to implement.
- **No Assumptions**: Doesn’t assume data distribution (non-parametric).
- **Versatility**: Works for classification (e.g., spam detection) and regression (e.g., house price prediction).
- **Adaptability**: Effective for small datasets and local patterns.

### Key Characteristics
- **Instance-Based Learning:** The algorithm compares new data points with existing data points based on similarity.
- **Non-Parametric**: Makes no assumptions about data distribution
- **Lazy Learner:** It does not learn from the training data immediately but stores it and uses it only when making a prediction.
- **Distance-Based Classification:** It assigns a new data point to the class that is most common among its `K` nearest neighbors.

---

## **How It Works**
1. **Training**: Store the training data (features \(X\) and labels \(y\)).

1. **Choose the Number of Neighbors (K):**  
   - Select a value for K, which determines how many neighbors influence the prediction.  
   - Small K (e.g., 1) can lead to noisy, overfitted predictions, while large K smooths the decision boundary but may underfit.

2. **Calculate Distances:**  
   - For a new data point, compute its distance to all points in the training set.  
   - Common distance metrics include:
     - **Euclidean Distance:** √((x₁ - x₂)² + (y₁ - y₂)²)  
     - **Manhattan Distance:** |x₁ - x₂| + |y₁ - y₂|  
     - **Minkowski Distance:** A generalization of Euclidean and Manhattan distances.

3. **Find the K Nearest Neighbors:**  
   - Sort the distances and select the K points with the smallest distances.

4. **Make a Prediction:**  
   - **Classification:** Use majority voting among the K neighbors’ labels.  
   - **Regression:** Compute the average (or weighted average) of the K neighbors’ values.

## **Hyperparameters**
- `n_neighbors` (\(k\)): Number of neighbors (small \(k\) = sensitive to noise, large \(k\) = smoother).
- `weights`: Uniform (equal weight) or distance-based (closer neighbors weigh more).
- `metric`: Distance measure (e.g., "euclidean", "manhattan").
- `p`: Power parameter for Minkowski distance (p=2 for Euclidean, p=1 for Manhattan).

---

## Choosing the Optimal K Value
### Key Considerations
- **Small K**: High variance, low bias (risk of overfitting)
- **Large K**: High bias, low variance (risk of underfitting)

### Selection Methods
1. **Elbow Method**: Plot accuracy vs K values
2. **Cross-Validation**: Use k-fold validation to find optimal K
3. **Square Root Rule**: K ≈ √n (n = training samples)

---
## Strengths of K-NN

- **Simple and Intuitive:** Easy to understand and implement.
- **No Training Phase:** Instantly usable since it stores data instead of building a model.
- **Versatile:** Works for both classification and regression.
- **Flexible:** Handles non-linear relationships well.

## Weaknesses of K-NN

- **Slow Predictions:** Computing distances for large datasets is time-consuming.
- **Memory Intensive:** Must store the entire training set.
- **Curse of Dimensionality:** Struggles in high-dimensional spaces where data becomes sparse.
- **Sensitive to Noise:** Irrelevant or noisy features can distort distance calculations.

---

## **3. Practical Examples**

### **Example 1: Classification (Iris Dataset)**

Let’s illustrate K-NN with the **Iris dataset**, a classic dataset containing 150 samples of iris flowers with four features (sepal length, sepal width, petal length, petal width) and three classes (Setosa, Versicolor, Virginica). We’ll implement K-NN in Python using scikit-learn.

### Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
```

### Step 2: Load and Prepare the Data

```python
# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (4 columns)
y = iris.target  # Labels (0, 1, 2 for the three classes)

# Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Step 3: Choose K and Train the Model

Let’s start with **K=3**.

```python
# Initialize the K-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model (stores the training data)
knn.fit(X_train, y_train)
```

### Step 4: Make Predictions and Evaluate

```python
# Predict labels for the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
```

**Sample Output:**
```
Accuracy: 0.98
Confusion Matrix:
 [[19  0  0]
  [ 0 13  0]
  [ 0  1 12]]
```
- Accuracy of 98% means the model correctly classified 44 out of 45 test samples.
- The confusion matrix shows one misclassification (a Virginica flower predicted as Versicolor).

---

## Tools and Methods for K-NN

To optimize K-NN, you can leverage several tools and techniques:

### 1. Feature Scaling
Since K-NN relies on distance, features with different scales (e.g., one in meters, another in kilometers) can skew results. Use **StandardScaler** or **MinMaxScaler** to normalize data.

```python
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain with scaled data
knn.fit(X_train_scaled, y_train)
y_pred_scaled = knn.predict(X_test_scaled)
print(f"Accuracy with Scaling: {accuracy_score(y_test, y_pred_scaled):.2f}")
```

### 2. Choosing the Optimal K
Test different K values to find the best one.

```python
# Test K from 1 to 20
accuracies = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot accuracy vs. K
plt.plot(range(1, 21), accuracies, marker='o')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. K')
plt.show()
```

This plot helps identify the K value that maximizes accuracy.

### 3. Distance Metrics
Experiment with different metrics (e.g., `'euclidean'`, `'manhattan'`) by setting the `metric` parameter in `KNeighborsClassifier`.

### 4. Weighted Voting
Give closer neighbors more influence by setting `weights='distance'` (default is `'uniform'`, where all neighbors have equal weight).

```python
knn_weighted = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_weighted.fit(X_train_scaled, y_train)
print(f"Accuracy with Weighted Voting: {accuracy_score(y_test, knn_weighted.predict(X_test_scaled)):.2f}")
```

### 5. Hyperparameter Tuning with GridSearchCV
Automate the search for the best K, weights, and metric.

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize and run GridSearchCV
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Results
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Test accuracy with best model
best_knn = grid_search.best_estimator_
print(f"Test Accuracy: {accuracy_score(y_test, best_knn.predict(X_test_scaled)):.2f}")
```

### 6. Cross-Validation
Evaluate model stability across data splits.

```python
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
print(f"Cross-Validation Scores: {scores}")
print(f"Average CV Score: {scores.mean():.2f}")
```

---

### **Example 2: Regression**
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Synthetic regression data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# Fit K-NN regression
knn_reg = KNeighborsRegressor(n_neighbors=5, weights="distance")
knn_reg.fit(X, y.ravel())

# Predict
y_pred = knn_reg.predict(X)
mse = mean_squared_error(y, y_pred)
print("MSE:", mse)

# Visualize
plt.scatter(X, y, label="Data", alpha=0.5)
plt.plot(X, y_pred, color="red", label="K-NN Fit")
plt.title("K-NN Regression (k=5)")
plt.legend()
plt.show()
```
---
## **4. Tools and Methods Summary**
- **Modeling**: `sklearn.neighbors.KNeighborsClassifier`, `KNeighborsRegressor`.
- **Evaluation**: `sklearn.metrics.accuracy_score`, `mean_squared_error`.
- **Tuning**: `sklearn.model_selection.GridSearchCV`.
- **Visualization**: `matplotlib.pyplot.contourf()`, `seaborn.scatterplot()`.

---

