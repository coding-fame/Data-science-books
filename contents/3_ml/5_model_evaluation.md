
# 4️⃣ Model Evaluation

**Model Evaluation Metrics**

---
# **Regression Metrics**
- **Mean Squared Error (MSE)**: Average of squared errors.
- **Root Mean Squared Error (RMSE)**: \(\sqrt{MSE}\).
- **Mean Absolute Error (MAE)**: Average of absolute errors.
- **R² (R-Squared)**: Proportion of variance explained by the model (0 to 1).

Common Cost Functions for Regression
In regression tasks, models predict continuous numerical values (e.g., house prices, temperatures). **Cost functions** quantify prediction accuracy by measuring the "distance" between predicted and actual values.  

✅ **Purpose of Cost Functions:**  
- Guide model training by providing feedback on prediction errors.  
- Enable optimization algorithms (e.g., Gradient Descent) to adjust model weights.  

## Distance-Based Error Explained  
For each data point:  
- **Actual Value**: $y$  
- **Predicted Value**: $y'$  
- **Error**: $y - y'$  

Cost functions aggregate these errors across the dataset to measure overall model performance. 

## 1. Mean Squared Error (MSE)
- MSE is the most widely used cost function in regression. 
- MSE is calculated as the **average of the squared differences** between predicted (y`) and actual (y) target values.
- Squaring the errors amplifies larger discrepancies, making MSE sensitive to outliers.

**Formula**: 
```math
MSE = (1/n) * Σ (yi - yi')²
```
Where:  
- **n** = Number of data points  
- **yi** = Actual target value  
- **yi'** = Predicted value  

**When to Use**:  
- When large errors should be penalized more heavily.
- Ideal for regression tasks where errors are assumed to be normally distributed.

**Example**:  
Suppose we’re predicting house prices. If a model predicts $300,000 for a house that actually costs $350,000, the squared error is:  
```math
(350,000 - 300,000)² = 2,500,000,000
``` 
Averaging such errors across all predictions yields the MSE.

### Python Implementation  
```python
from sklearn.metrics import mean_squared_error

# Sample data: Actual vs. Predicted values
expected = [1.0] * 11
predicted = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

mse = mean_squared_error(expected, predicted)
print(f"MSE: {mse:.2f}")  # Output: MSE: 0.35
```

---
## 2. Mean Absolute Error (MAE) 
- MAE is calculated as the **average absolute difference** between predicted and actual values.
- Unlike MSE, it doesn’t square errors, making it less sensitive to outliers.

**Formula**:  
```math
MAE = (1/n) * Σ |yi - yi'|
```
**When to Use**:  
- When robustness to outliers is desired.
- Suitable when all errors should be weighted equally, regardless of magnitude.

**Example**:  
Using the same house price scenario:  
```math
|350,000 - 300,000| = 50,000
```
MAE averages these absolute differences, providing a simple measure of average error magnitude.

### Python Implementation  
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(expected, predicted)
print(f"MAE: {mae:.2f}")  # Output: MAE: 0.50
```

---
## 3. Root Mean Squared Error (RMSE) 
- RMSE is the square root of MSE, returning the error to the same units as the target variable. 
- It retains MSE’s sensitivity to larger errors but is more interpretable.

**Formula**:  
```math
RMSE = sqrt(MSE)
```
**When to Use**:  
- When you need an error metric in the same units as the target variable.
- Often used to report model performance in an intuitive way.

**Example**:  
If MSE is 2,500,000,000 (from the earlier example), then:  
```math
RMSE = sqrt(2,500,000,000) = 50,000
```
This indicates an average error of $50,000 in house price predictions.

### Python Implementation  
```python
rmse = mean_squared_error(expected, predicted, squared=False)
print(f"RMSE: {rmse:.2f}")  # Output: RMSE: 0.59
```

---
## Tools and Methods

Several Python libraries simplify the implementation of these cost functions:
- **Scikit-learn**: Provides `mean_squared_error`, `mean_absolute_error`, and supports custom extensions.
- **TensorFlow**: Offers `tf.keras.losses` with `MeanSquaredError`, `MeanAbsoluteError`, `Huber`, etc.
- **PyTorch**: Includes `torch.nn` modules like `MSELoss`, `L1Loss` (MAE), and `SmoothL1Loss` (Huber-like).

---

## One-Stop Solution: Python Code Example

This script uses the **California Housing dataset** to train a linear regression model and compute MSE, MAE, RMSE, Huber Loss, and Log-Cosh Loss.

```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 1. Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# 2. Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# 3. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# 4. Huber Loss (using TensorFlow)
huber = tf.keras.losses.Huber(delta=1.0)
huber_loss = huber(y_test, y_pred).numpy()
print(f"Huber Loss (delta=1.0): {huber_loss:.4f}")

# 5. Log-Cosh Loss (manual implementation)
log_cosh_loss = np.mean(np.log(np.cosh(y_test - y_pred)))
print(f"Log-Cosh Loss: {log_cosh_loss:.4f}")
```

---

## Key Takeaways  
1️⃣ **MSE**: Use when large errors must be flagged (e.g., fraud detection).  
2️⃣ **RMSE**: Default choice for model comparison and reporting.  
3️⃣ **MAE**: Best for noisy datasets or when outliers should not dominate error calculations.  

🔍 **Pro Tip**: Always visualize residuals (prediction errors) to understand error distribution and metric suitability!

---

# **Classification Metrics**
- **Accuracy**: (Correct Predictions) / (Total Predictions).
- **Precision**: (True Positives) / (True Positives + False Positives).
- **Recall**: (True Positives) / (True Positives + False Negatives).
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the ROC curve (measures class separation).
- **Confusion Matrix**: Table showing true vs. predicted classes.

---

Common Cost Functions for Classification

## 1. Binary Cross-Entropy (Log Loss)
- Binary cross-entropy measures the difference between predicted probabilities and true binary labels (0 or 1). 
- It’s widely used in binary classification tasks and penalizes predictions more heavily as they deviate from the true label.

**Formula:**  
For a single sample:  
BCE = - [y * log(ŷ) + (1 - y) * log(1 - ŷ)]  

For a dataset with *n* samples:  
BCE = - (1/n) * Σ [yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ)], for i = 1 to n  

Where:  
- *yᵢ* = True label (0 or 1)  
- *ŷᵢ* = Predicted probability for the positive class (between 0 and 1)  

**When to Use:**  
- Binary classification problems (e.g., spam vs. not spam).  
- Models that output probabilities (e.g., logistic regression).  

**Example:**  
- If *y = 1* and *ŷ = 0.9*:  
  BCE = - [1 * log(0.9) + (1 - 1) * log(1 - 0.9)]  
       = - log(0.9) ≈ 0.105  

- If *y = 1* and *ŷ = 0.1*:  
  BCE = - log(0.1) ≈ 2.303  

The loss increases significantly for poor predictions.

---

## 2. Categorical Cross-Entropy 
- Categorical cross-entropy extends binary cross-entropy to multi-class classification. 
- It compares the true label distribution (typically one-hot encoded) with the predicted probability distribution across all classes.

**Formula:**  

For a single sample with C classes:  
`CCE = - Σ (y_c * log(ŷ_c))`  

For a dataset:  
`CCE = - (1/n) * Σ Σ (y_i,c * log(ŷ_i,c))`  

Where:  
- y_i,c = 1 if the sample belongs to class c, otherwise 0.  
- ŷ_i,c = Predicted probability for class c.  

---

### **When to Use**  
- Multi-class classification (e.g., classifying images into 10 digit categories).  
- Models that output probability distributions (e.g., softmax output).  

### **Example**  
For a 3-class problem:  
- True label = class 2 (one-hot: [0, 1, 0])  
- Predicted probabilities = [0.1, 0.7, 0.2]  

Calculation:  
`CCE = - (0 * log(0.1) + 1 * log(0.7) + 0 * log(0.2))`  
`CCE = - log(0.7) ≈ 0.357`  

---

## 3. Sparse Categorical Cross-Entropy
- Similar to categorical cross-entropy, but designed for integer labels instead of one-hot encoded vectors. 
- It’s more memory-efficient, especially with many classes.

**Formula**:  
For a single sample:  
Sparse CCE = - log(ŷ_y)  

Where:  
- y = True class index  
- ŷ_y = Predicted probability for the true class  

**When to Use**  
- Multi-class classification with integer labels  
- Large number of classes where one-hot encoding is impractical  

**Example**  
True label = class 2 (index 2)  
Predicted probabilities = [0.1, 0.7, 0.2]  

Calculation:  
`Sparse CCE = - log(0.7) ≈ 0.357` 

---

## 4. Hinge Loss
- Hinge loss is used in Support Vector Machines (SVMs) for binary classification. 
- It encourages correct classification with a margin, penalizing predictions that are correct but too close to the decision boundary.

For a single sample:  
Hinge = max(0, 1 - y * ŷ)  

Where:  
- y = True label (-1 or 1)  
- ŷ = Predicted score (not a probability)  

**When to Use**  
- SVM-based classification  
- Models outputting decision scores rather than probabilities  

**Example**  
1. If y = 1, ŷ = 0.8:  
   Hinge = max(0, 1 - (1 * 0.8)) = max(0, 0.2) = **0.2**  

2. If y = 1, ŷ = -0.5:  
   Hinge = max(0, 1 - (1 * -0.5)) = max(0, 1.5) = **1.5**  

---

## Tools and Methods

Python libraries provide built-in implementations of these cost functions:
- **Scikit-learn**: `log_loss` (cross-entropy), `hinge_loss`.
- **TensorFlow**: `tf.keras.losses` (e.g., `BinaryCrossentropy`, `CategoricalCrossentropy`, `SparseCategoricalCrossentropy`, `Hinge`).
- **PyTorch**: `torch.nn` (e.g., `CrossEntropyLoss`, `BCELoss`).
- **NumPy**: For manual implementations or custom loss functions.

---

## One-Stop Solution: Python Code Example

This script uses the **Iris dataset** for multi-class classification and a synthetic dataset for binary classification, computing the cost functions discussed.

```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# --- Binary Classification ---
# Generate synthetic binary data
X_bin, y_bin = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

# Train logistic regression
model_bin = LogisticRegression()
model_bin.fit(X_train_bin, y_train_bin)
y_pred_prob_bin = model_bin.predict_proba(X_test_bin)[:, 1]  # Probability of class 1

# 1. Binary Cross-Entropy
bce = log_loss(y_test_bin, y_pred_prob_bin)
print(f"Binary Cross-Entropy (Log Loss): {bce:.4f}")

# 2. Hinge Loss (using SVM)
model_svm = SVC(kernel='linear')
model_svm.fit(X_train_bin, y_train_bin)
y_pred_decision = model_svm.decision_function(X_test_bin)  # Decision scores
y_test_bin_svm = 2 * y_test_bin - 1  # Convert 0/1 to -1/1
hinge_loss = np.mean(np.maximum(0, 1 - y_test_bin_svm * y_pred_decision))
print(f"Hinge Loss: {hinge_loss:.4f}")

# --- Multi-Class Classification ---
# Load Iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Train logistic regression
model_multi = LogisticRegression(multi_class='ovr', max_iter=200)
model_multi.fit(X_train_iris, y_train_iris)
y_pred_prob_multi = model_multi.predict_proba(X_test_iris)

# 3. Categorical Cross-Entropy
cce = log_loss(y_test_iris, y_pred_prob_multi)
print(f"Categorical Cross-Entropy: {cce:.4f}")

# 4. Sparse Categorical Cross-Entropy (TensorFlow)
sparse_cce = tf.keras.losses.SparseCategoricalCrossentropy()
sparse_cce_loss = sparse_cce(y_test_iris, y_pred_prob_multi).numpy()
print(f"Sparse Categorical Cross-Entropy: {sparse_cce_loss:.4f}")

# 5. Focal Loss (manual implementation for binary)
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    return -alpha * (1 - pt) ** gamma * np.log(pt + 1e-10)  # Add small constant to avoid log(0)

focal_loss_value = np.mean([focal_loss(y, p) for y, p in zip(y_test_bin, y_pred_prob_bin)])
print(f"Focal Loss (gamma=2, alpha=0.25): {focal_loss_value:.4f}")
```

### Code Explanation
- **Binary Classification**:
  - Uses `make_classification` to create synthetic data.
  - Computes **Binary Cross-Entropy** with `log_loss`.
  - Computes **Hinge Loss** manually using SVM decision scores (labels converted to -1/1).
- **Multi-Class Classification**:
  - Uses the Iris dataset.
  - Computes **Categorical Cross-Entropy** with `log_loss`.
  - Computes **Sparse Categorical Cross-Entropy** with TensorFlow.
- **Focal Loss**:
  - Manually implemented for the binary case, adding a small constant to prevent log(0) errors.

### Sample Output (Approximate)
```
Binary Cross-Entropy (Log Loss): 0.2456
Hinge Loss: 0.1234
Categorical Cross-Entropy: 0.1234
Sparse Categorical Cross-Entropy: 0.1234
Focal Loss (gamma=2, alpha=0.25): 0.0567
```
*(Actual values vary based on data splits and model performance.)*

## Conclusion

Cost functions are critical for training classification models:
- **Binary Cross-Entropy**: Ideal for binary tasks with probability outputs.
- **Categorical Cross-Entropy**: Suited for multi-class problems with one-hot labels.
- **Sparse Categorical Cross-Entropy**: Efficient for multi-class integer labels.
- **Hinge Loss**: Best for SVMs and margin maximization.
- **Focal Loss**: Effective for imbalanced datasets by focusing on hard examples.

---

## Conclusion
Cost functions are indispensable in machine learning, serving as the foundation for evaluating and optimizing models. 

---

# **Clustering Metrics**
- Silhouette Score, Davies-Bouldin Index.

---

# Confusion Matrix

A **Confusion Matrix** is a powerful tool used in machine learning to evaluate the performance of classification models.

---

## **What is a Confusion Matrix?**
A confusion matrix is a tabular representation of a classification model’s performance, comparing predicted labels to actual labels. It summarizes the counts of correct and incorrect predictions across all classes, providing a detailed breakdown beyond simple accuracy.

## **Why Use a Confusion Matrix?**
- **Granular Insight**: Reveals specific errors (e.g., false positives vs. false negatives).
- **Class Imbalance**: Highlights performance in imbalanced datasets where accuracy alone is misleading.
- **Derived Metrics**: Basis for precision, recall, F1-score, and more.
- **Decision Making**: Helps assess model suitability for specific tasks (e.g., minimizing false negatives in medical diagnosis).

## **Structure**
For a binary classification problem (positive vs. negative):
- **True Positive (TP)**: The model correctly predicts the positive class (e.g., identifying a sick patient as sick).
- **True Negative (TN)**: The model correctly predicts the negative class (e.g., identifying a healthy patient as healthy).
- **False Positive (FP)**: The model incorrectly predicts the positive class (a Type I error). (e.g., saying a healthy patient is sick)
- **False Negative (FN)**: The model incorrectly predicts the negative class (a Type II error). (e.g., saying a sick patient is healthy)


|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | TP                | FN                |
| **Actual Negative** | FP                | TN                |

For multiclass, it extends to a \(k \times k\) matrix where \(k\) is the number of classes.

---
## Example of a Confusion Matrix

Let’s consider a binary classification problem: predicting whether an email is **spam** (positive class) or **not spam** (negative class). Suppose a model makes predictions on 175 emails, with the following results:

- **TP**: 50 spam emails correctly classified as spam.
- **TN**: 100 non-spam emails correctly classified as non-spam.
- **FP**: 15 non-spam emails incorrectly classified as spam.
- **FN**: 10 spam emails incorrectly classified as non-spam.

The confusion matrix would look like this:

|           | Predicted Spam | Predicted Not Spam |
|-----------|----------------|--------------------|
| **Actual Spam**    | 50 (TP)        | 10 (FN)            |
| **Actual Not Spam**| 15 (FP)        | 100 (TN)           |

This table shows the model correctly classified 150 emails (TP + TN = 50 + 100) and misclassified 25 (FP + FN = 15 + 10).

---

## Metrics Derived from the Confusion Matrix

The confusion matrix enables the calculation of several key performance metrics:

### **Components**
- **Diagonal**: Correct predictions (TP for each class in multiclass).
- **Off-Diagonal**: Errors (FP and FN for binary; misclassifications for multiclass).

### **1. Accuracy**
- **What It Means**: The percentage of correct predictions.
- **Formula**:
  ```python
  accuracy = (TP + TN) / (TP + FP + FN + TN)
  ```
- **Example**: If TP = 50, TN = 105, FP = 10, FN = 15, then:
  ```python
  accuracy = (50 + 100) / (50 + 100 + 15 + 10)
  accuracy = 150/175 # 0.857 --> 85.7%
  ```
- **When to Use**: Good for balanced data, but misleading if one class dominates.

### **2. Precision**
- **What It Means**: How many predicted positives are actually correct.
- **Formula**:
  ```python
  recall = TP / (TP + FP)  # "How many did we catch?"
  ```
- Interpretation: The proportion of predicted positives that are actually positive (how precise the positive predictions are).
- **Example**: 
  ```python
  recall = 50 / (50 + 15)  # 76.9%
  ```
- **When to Use**: Important when false positives are costly (e.g., cancer screening).

### **3. Recall (Sensitivity)**
- **What It Means**: How many actual positives the model finds.
- **Formula**:
  ```python
  recall = TP / (TP + FN)  # "How many did we catch?"
  ```
- Interpretation: The proportion of actual positives correctly identified (how well the model captures positives).
- **Example**: 
  ```python
  recall = 50 / (50 + 10)  # 83%
  ```
- **When to Use**: Critical when missing positives is risky (e.g., detecting diseases).

### **4. F1 Score**
- **What It Means**: A balance between precision and recall.
- **Formula**:
  ```python
  F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
  ```
- Interpretation: The harmonic mean of precision and recall, balancing both when they’re equally important.
- **Example**: 
  ```python
  F1 Score = 2 * (0.769 * 0.833) / (0.769 + 0.833) # 80%
  ```
- **When to Use**: When you want a single score to compare models.

### **5. Specificity**
- Formula:  
    ```python
    Specificity = TN / (TN + FP)
    ```
- Interpretation: The proportion of actual negatives correctly identified.
- Example: 
  ```python
  F1 Score = 100 / (100 + 15) # 87%
  ```

These metrics provide a comprehensive view of the model's performance, tailored to different priorities (e.g., minimizing false positives vs. maximizing true positives).

---

## One-Stop Solution: Python Code Example

This example uses the **Iris dataset** (a multi-class classification problem with three classes: setosa, versicolor, virginica) to train a model, compute the confusion matrix, visualize it, and calculate metrics.

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # Features and target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # Make predictions on the test set

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Iris Dataset")
plt.show()

# Print a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### Code Explanation

1. **Data Loading and Splitting**:
   - The Iris dataset is loaded with 150 samples across 3 classes.
   - It’s split into 80% training and 20% testing data (`test_size=0.2`).

2. **Model Training**:
   - A `RandomForestClassifier` is trained on the training data.

3. **Confusion Matrix Computation and Visualization**:
   - `confusion_matrix` computes the matrix comparing `y_test` (actual labels) and `y_pred` (predicted labels).
   - `ConfusionMatrixDisplay` creates a heatmap of the matrix, labeled with class names (setosa, versicolor, virginica).

4. **Classification Report**:
   - `classification_report` outputs precision, recall, F1-score, and support (number of samples) for each class, plus overall averages.

### Sample Output

#### Confusion Matrix Plot
A 3x3 heatmap might look like this (values depend on the random split):

```
           Predicted
           | Setosa | Versicolor | Virginica |
Actual     |--------|------------|-----------|
Setosa     |   10   |     0      |     0     |
Versicolor |    0   |     9      |     1     |
Virginica  |    0   |     1      |     9     |
```

- Diagonal values (10, 9, 9) are TPs for each class.
- Off-diagonal values (0s and 1s) are FPs and FNs.

#### Classification Report
```
Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10
    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30
```

- **Setosa**: Perfectly classified (precision, recall, F1 = 1.00).
- **Versicolor/Virginica**: Minor errors (e.g., one misclassification each), yielding 0.90 for precision, recall, and F1.
- **Accuracy**: 93% overall.

*(Note: Exact numbers may vary due to randomness in the split and model.)*

---
## **AUC and ROC Curve**

The **Area Under the Curve (AUC)** measures how well the model separates classes. It comes from the **ROC Curve**, which plots Recall (True Positive Rate) against False Positive Rate.

- **AUC = 1**: Perfect model.
- **AUC = 0.5**: No better than guessing.
- **AUC < 0.5**: Worse than random.

### **Code Example**

```python
from sklearn.metrics import roc_curve, auc

# Get prediction probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot it
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Titanic Survival')
plt.legend()
plt.show()
```

A higher AUC means better performance.

---

## **Tools and Methods Summary**
- `sklearn.metrics.confusion_matrix`: Computes the confusion matrix from true and predicted labels.
- `sklearn.metrics.classification_report`: Provides a detailed report with precision, recall, F1-score, and support.
- **Visualization**: `seaborn.heatmap()`, `matplotlib.pyplot` for custom plots.

```python
# Custom metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
```



