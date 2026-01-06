
# Regression Algorithms

---

# ML: R-Value and Regression Analysis

## 🔍 Introduction to Regression
Regression analysis is a fundamental concept in Machine Learning used to explain the relationship between a **dependent variable** and one or more **independent variables**.
  
---

## 📈 Understanding the Regression Line
- If two variables are related, we can visualize their relationship in a **two-dimensional space**.
- The result is often a **straight line** that represents their correlation.
- Think of it as plotting scattered points and finding the best possible line to pass through them.

---

### 🎯 The Goal of Regression
- The primary objective of **Linear Regression** is to draw the **best-fit line**.
- A **best-fit line** is the one that passes as **close as possible** to all data points.

---

## 🤔 When to Use Regression?

**Linear Regression** can only be applied when there is a clear relationship between the variables.

| Scenario | Can We Use Regression? |
|----------|----------------------|
| There is a **strong relationship** between variables | ✅ Yes |
| There is **no significant relationship** between variables | ❌ No |

---

## 📌 What is R Value?
- **R value (correlation coefficient)** helps measure how strongly two variables are related.
- It is a critical step in determining whether **Linear Regression** can be applied to a dataset.

### 📏 R Value Range Interpretation
| R Value | Interpretation |
|---------|---------------|
| **1.0 or -1.0** | Strong relationship between variables (✅ Regression is applicable) |
| **Close to 0** | Weak or no relationship (❌ Regression is not applicable) |

---

## 🧑‍💻 Calculating R Value in Python
We can use the **scipy.stats** module to compute the R value.

```python
from scipy import stats

# Example dataset
X = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Calculate regression statistics
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

print("R Value:", r_value)  # Output: 1.0 (Strong Relationship)
```

---

## 📊 Example Interpretations
### ✅ Strong Relationship (Regression is useful)
- **Result**: `R Value = 1.0`
- **Conclusion**: A strong relationship exists → **Linear Regression can be applied** for future predictions.

### ❌ Weak or No Relationship (Regression is not useful)
- **Result**: `R Value = -0.065`
- **Conclusion**: The variables are **not related** → **Linear Regression will give inaccurate predictions**.

---

## Summary
- **Linear regression** is a powerful tool, but it’s essential to first check if there’s a **relationship** between the variables.
- The **r value** helps in assessing the strength of this relationship, guiding whether regression is the right approach.

---


# Linear Regression

Linear regression is a fundamental algorithm used in **supervised learning** for predicting continuous values.

## Introduction to Regression
**Regression analysis** is used to understand the relationship between two variables. It is widely used in data science and machine learning for **predictive modeling**.

# What Is Linear Regression?

**Linear Regression** is a fundamental machine learning technique used to model the relationship between **dependent** and **independent** variables. It helps in predicting outcomes based on input data.

Linear regression assumes that the relationship between variables can be represented by a straight line (or a flat surface in higher dimensions). It’s widely used because it’s simple, effective, and provides a strong foundation for more advanced ML techniques.

### Types of Linear Regression
There are two main types:

### Simple Linear Regression
- Uses **one independent variable** (X) to predict **one dependent variable** (Y).
- The goal is to draw a **straight line** that best fits the data, showing how changes in the independent variable affect the dependent variable.
- Example: Predicting house price based on area.
- Formula:
  ```
  y = m * X + b
  ```
  Where:  
  - **y** = Dependent variable (Predicted output)  
  - **X** = Independent variable (Input feature)  
  - **m** = Slope (Coefficient)  
  - **b** = Intercept (Constant)

This formula creates a straight line, where **m** controls the tilt and **b** shifts the line up or down.

### Multiple Linear Regression
- Uses **two or more independent variables** (X₁, X₂, … Xₙ) to predict **one dependent variable** (Y).
- The goal is to find the coefficients that define the best-fitting **hyperplane**, minimizing the sum of squared differences between actual and predicted values.
- Example: Predicting house price based on **area, location, and number of bedrooms**.
- Formula:  
  ```
  y = m₁*X₁ + m₂*X₂ + .... + mₙ*Xₙ + b
  ```  
  Where:
  - **Y** = Predicted price
  - **X₁, X₂, X₃** = Independent variables (Area, Bedrooms, Age)
  - **m₁, m₂, m₃** = Coefficients (Slope values) (the effect of each x on y)
  - **b** = Intercept (Bias)

This formula creates a straight line, where **m** controls the tilt and **b** shifts the line up or down.

---

## Key Concepts

### 1. Coefficients
- **Slope (m)**: Measures the impact of the independent variable. For example, if m = 135.79, the price increases by $135.79 for each square foot.
- **Intercept (b)**: The starting point of the line when x = 0.
- The **intercept (b)** and **coefficients (M₁, M₂, M₃)** define the linear equation.

### 2. Assumptions
Simple linear regression works best when:
- The relationship between x and y is **linear** (a straight line fits).
- Data points are **independent** (one doesn’t affect another) that is uncorrelated (No Multicollinearity).
- Errors (differences between actual and predicted values) have **constant variance** (Homoscedasticity) and are **normally distributed**.

### 3. Evaluation Metrics
To check how good the model is:
- **Mean Squared Error (MSE)**: Average of squared differences between actual and predicted values. Lower is better.
- **R-squared (R²)**: Shows how much of the data’s variation the model explains (0 to 1, closer to 1 is better).

### 4. Best Fit line
- The "best-fitting" line is the one that gets as close as possible to all data points. 
- In ML, this is done by minimizing the **sum of squared errors (SSE)** the total difference between actual and predicted values.
- The model finds the best-fit line that **minimizes the prediction error**.

## Limitations
- Assumes linearity; struggles with complex non-linear relationships.
- Sensitive to outliers and multicollinearity.
- Requires careful validation of assumptions.

## Handling Violations of Assumptions
- **Non-linearity**: Add polynomial/interaction terms.
- **Heteroscedasticity**: Use weighted least squares or transform Y.
- **Non-normality**: Transform Y (e.g., log transformation).
- **Multicollinearity**: Remove correlated variables or use regularization (Ridge/Lasso).

### Regularization Techniques
- Ridge Regression (L2): Shrinks coefficients but does not eliminate them.
- Lasso Regression (L1): Performs feature selection by driving some coefficients to zero.

---

# Simple Linear: Predicting House Price

Let’s use simple linear regression to predict home prices based on their area. Here, **area** is the independent variable (x), and **price** is the dependent variable (y).

## Step 1: Set Up the Tools
We’ll use Python and the **Scikit-Learn** library, a popular tool in ML, to build our model.

```python
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
```

## Step 2: Prepare the Data
Imagine we have this data:

| Area (sq. ft) | Price ($) |
|---------------|-----------|
| 1500          | 300,000   |
| 1800          | 350,000   |
| 2000          | 400,000   |
| 2200          | 420,000   |
| 2500          | 480,000   |

We’ll store it in a table (DataFrame) using Python:
```python
data = {'area': [1500, 1800, 2000, 2200, 2500],
        'price': [300000, 350000, 400000, 420000, 480000]}
df = pd.DataFrame(data)
```

## Step 3: Train the Model
We create a linear regression model and train it with our data:
```python
reg = LinearRegression()
reg.fit(df[['area']], df['price'])
```
- `fit()` finds the best **m** and **b** values to match the data.

## Step 4: Make a Prediction
Let’s predict the price for a house with 3300 sq. ft:
```python
prediction = reg.predict([[3300]])
print("Predicted price:", prediction)
```
This gives us the estimated price based on the trained model.

## Step 5: Check the Coefficients
The model calculates:
- **Slope (m)**: `reg.coef_` (e.g., 135.79)
- **Intercept (b)**: `reg.intercept_` (e.g., 180616.44)

```python
print("Slope (m):", reg.coef_)
print("Intercept (b):", reg.intercept_)
```

## Step 6: Manual Calculation
Using the formula `y = m * x + b`:
```python
m = reg.coef_[0]
b = reg.intercept_
x = 3300
y_manual = m * x + b
print("Manual prediction:", y_manual)
```
This should match the result from `reg.predict()`.

---

## Visualize the Fit
We can plot the data and the line:
```python
plt.xlabel('Area (sq. ft)')
plt.ylabel('Price ($)')
plt.scatter(df['area'], df['price'], color='red', marker='*')
plt.plot(df['area'], reg.predict(df[['area']]), color='blue')
plt.show()
```

- **Red stars**: Actual data points.
- **Blue line**: Predicted values (the best-fitting line).

This graph shows how well the line fits the data.

---

# Example: Multiple Linear Regression

## Problem Statement
Imagine you're planning to buy a new house and need to predict the price based on multiple factors:
- **Area (square feet)**
- **Number of bedrooms**
- **Age of the house (in years)**

Given the home price dataset, we aim to predict the price of homes with:
- **3000 sq. ft area, 3 bedrooms, 40 years old**
- **2500 sq. ft area, 4 bedrooms, 5 years old**

## Dataset Overview
We will use **homeprices1.csv** as our dataset, which contains the following columns:
- **Area** (Square Feet)
- **Bedrooms** (Number of Bedrooms)
- **Age** (Years)
- **Price** (Target Variable)

## Implementing Multiple Linear Regression in Python
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("homeprices1.csv")

# Handle missing values by filling with median of 'bedrooms'
df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)

# Separate features and target variable
X = df[['area', 'bedrooms', 'age']]
y = df['price']

# Train the model
reg = LinearRegression()
reg.fit(X, y)

# Display model parameters
print("Intercept:", reg.intercept_)
print("Coefficients:", reg.coef_)

# Predict prices for new homes
new_homes = pd.DataFrame({
    'area': [3000, 2500],
    'bedrooms': [3, 4],
    'age': [40, 5]
})

predicted_prices = reg.predict(X)
print("Predicted Prices:", predicted_prices)

print(f"R²: {r2_score(y, predicted_prices)}, RMSE: {np.sqrt(mean_squared_error(y, predicted_prices))}")
```

## Making Predictions
To predict house prices for new homes:
```python
predicted_price = reg.predict([[3000, 3, 40]])
print(f"Predicted price for 3000 sq.ft, 3 bedrooms, 40 years old: ${predicted_price[0]:,.2f}")

predicted_price = reg.predict([[2500, 4, 5]])
print(f"Predicted price for 2500 sq.ft, 4 bedrooms, 5 years old: ${predicted_price[0]:,.2f}")
```

---

# **Example: Predicting California House Prices**

We’ll use the **California housing dataset** to predict median house values based on features like median income, house age, and average rooms.

---

## **1. Data Collection** 🗂️

**Objective**: Gather relevant data for the machine learning task.  
**Description**: For this example, we’ll use the California housing dataset, which is available through Scikit-Learn. This dataset includes features such as median income (`MedInc`), house age (`HouseAge`), average number of rooms (`AveRooms`), and others, with the target variable being the median house value (`MedHouseVal`).  
**Code**:
```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target
```
**Explanation**: The `fetch_california_housing()` function retrieves the dataset, and we convert it into a Pandas DataFrame for easier manipulation. The features are stored in `data.data`, and the target variable is in `data.target`. This step simulates collecting data from a reliable source.

---

## **2. Data Preparation** 🔍

**Objective**: Clean and preprocess the data to make it suitable for modeling.  
**Actions**:
- Check for and handle missing values.
- Select relevant features for the model.
- Scale the features to standardize them for better model performance.  

**Code**:
```python
# Check for missing values
print(df.isnull().sum())  # No missing values in this dataset

# Select features and target
X = df[['MedInc', 'HouseAge', 'AveRooms']]
y = df['MedHouseVal']

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**Explanation**: 
- **Missing Values**: The California housing dataset is clean and has no missing values, as confirmed by `isnull().sum()`.
- **Feature Selection**: We choose three features—`MedInc` (median income), `HouseAge` (median house age), and `AveRooms` (average rooms)—to predict the target `MedHouseVal`.
- **Scaling**: `StandardScaler` standardizes the features by removing the mean and scaling to unit variance, which helps the model converge faster and perform better.

---

## **3. Data Wrangling** 🛠️

**Objective**: Structure and split the data for training and testing.  
**Action**: Divide the dataset into training and testing sets to evaluate the model on unseen data.  
**Code**:
```python
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```
**Explanation**: We use `train_test_split` to allocate 80% of the data for training (`X_train`, `y_train`) and 20% for testing (`X_test`, `y_test`). The `random_state=42` ensures reproducibility. This step prepares the data for model training and evaluation.

---

## **4. Train the Model** 🎯

**Objective**: Build a predictive model using the training data.  
**Action**: Train a multiple linear regression model on the selected features.  
**Code**:
```python
from sklearn.linear_model import LinearRegression

# Initialize and train the model
reg = LinearRegression()
reg.fit(X_train, y_train)
```
**Explanation**: The `LinearRegression` model from Scikit-Learn is used to fit a linear equation to the training data. The `fit` method computes the optimal coefficients for the features (`MedInc`, `HouseAge`, `AveRooms`) to predict `MedHouseVal`.

---

## **5. Test the Model** 🧪

**Objective**: Evaluate the model’s performance on the test data.  
**Actions**:
- Generate predictions for the test set.
- Calculate evaluation metrics such as R-squared and Mean Squared Error (MSE).  

**Code**:
```python
from sklearn.metrics import r2_score, mean_squared_error

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R-squared:", r2)
print("MSE:", mse)
```
**Explanation**: 
- **Predictions**: The `predict` method uses the trained model to estimate house values for the test set.
- **Metrics**: 
  - **R-squared**: Measures the proportion of variance in the target variable explained by the model (closer to 1 is better).
  - **MSE**: Calculates the average squared difference between predicted and actual values (lower is better).  
These metrics indicate how well the model generalizes to unseen data.

---

## **6. Model Deployment** 🚀

**Objective**: Make the trained model available for real-world use.  
**Action**: Save the model to a file for later use or deployment.  
**Code**:
```python
import joblib

# Save the trained model
joblib.dump(reg, 'california_housing_model.pkl')
```
**Explanation**: The `joblib.dump` function serializes the trained model to a file named `california_housing_model.pkl`. This file can be loaded later to make predictions without retraining. In a real-world scenario, you might deploy this model via a web service (e.g., using Flask or FastAPI), but saving it is a key first step.

---

# ML: Polynomial Features

Polynomial features extend linear regression by adding terms that are powers or interactions of the original features, allowing the model to fit non-linear patterns. While basic linear regression assumes a straight-line relationship (\(y = w_0 + w_1x\)), polynomial regression introduces higher-degree terms (e.g., \(x^2, x^3\)) and interactions (e.g., \(x_1x_2\)) to model curves and complex dependencies.

---

## Overview
Polynomial Features are a type of **feature engineering** where we create new input features based on existing ones by applying polynomial transformations.

### Example
If we have a dataset with one input feature \( X \), we can create a new feature by squaring \( X \), i.e., \( X^2 \). This process can be extended for higher-degree polynomials:
- **Degree 1:** \( X \)
- **Degree 2:** \( X, X^2 \)
- **Degree 3:** \( X, X^2, X^3 \), etc.

The **degree** of the polynomial controls the number of new features added.

---

## Why Do We Need Polynomial Features?

### Case 1: Linear Data
- If we apply a **linear model** to **linear data**, it works well (as seen in **Simple Linear Regression**).
- However, if the dataset is **non-linear**, using a linear model **without modification** will produce **poor results**.
- This leads to **high error rates** and inaccurate predictions.

### Case 2: Non-Linear Data
- If we use a linear model on **non-linear data**, the predictions will be incorrect.
- This leads to **high error rates** and poor model performance.
- **Solution:** Use **Polynomial Regression**, which extends linear regression by adding polynomial terms.

---

### **Why Use Polynomial Features?**
- **Non-Linearity**: Real-world data often exhibits non-linear relationships (e.g., quadratic growth).
- **Flexibility**: Adds expressive power to linear models without changing the algorithm.
- **Feature Engineering**: Enhances model performance by capturing more patterns.

## **Mathematical Representation**

Simple Linear Regression Equation
```math
y = b_0 + b_1x
```

For a single feature \(x\), polynomial regression of degree \(n\) might look like:
```math
y = b_0 + b_1x + b_2x^2 + b_3x^3 + ... + b_nx^n
```
For multiple features (e.g., \(x_1, x_2\)), it includes interaction terms:
```math
y = b_0 + b_1x_1 + b_2x_2 + b_3x_1^2 + b_4x_1x_2 + b_5x_2^2 + \cdots
```

## Step : Simple Linear
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("poly_dataset.csv")

# Extracting independent and dependent variables
X = df.iloc[:, 1:2].values  # Selecting feature column
y = df.iloc[:, 2].values    # Target variable

# Scatter plot to visualize data distribution
plt.scatter(X, y, color="blue")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.title("Data Distribution")
plt.show()

# Train a simple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Plot the Linear Regression model
plt.scatter(X, y, color="blue")
plt.plot(X, lin_reg.predict(X), color="red")
plt.title("Linear Regression Fit")
plt.show()
```

## **Key Concepts and Methods**

### **a. Generating Polynomial Features**
- **Concept**: Transform original features into a higher-dimensional space with polynomial terms.
- **Parameters**:
  - `degree`: Maximum power of features.
  - `interaction_only`: Include only interaction terms (e.g., \(x_1x_2\)).
  - `include_bias`: Add a constant term (intercept).
```python
# Polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

print("Original X:\n", X)
print("Polynomial X (degree 2):\n", X_poly)
print("Feature Names:", poly.get_feature_names_out())
```

### **b. Polynomial Regression**
- **Concept**: Fit a linear model to the polynomial features.
- **Steps**: Transform data, then apply linear regression.
- **`y.ravel()` vs. `y`**  
  If `y` is already a 1D array, `.ravel()` has no effect. If `y` is a column vector (e.g., from a Pandas DataFrame), `.ravel()` reshapes it to avoid warnings/errors.

```python
from sklearn.linear_model import LinearRegression

# Target: Quadratic relationship y = 2x^2 + x + 1
X = np.array([[0], [1], [2], [3]])
y = 2 * X**2 + X + 1

# Transform
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit model
model = LinearRegression()
model.fit(X_poly, y.ravel())
print("Coefficients:", model.coef_)  # Output: [0, 1, 2] (bias adjusted in intercept)
print("Intercept:", model.intercept_)  # Output: ~1

plt.scatter(X, y, color="blue")
plt.plot(X, poly_model.predict(X_poly), color="red")
plt.xlabel("Input Feature")
plt.ylabel("Target Variable")
plt.title("Polynomial Regression Fit")
plt.show()
```

### **c. Interaction Terms**
- **Concept**: Capture relationships between features (e.g., \(x_1x_2\)).

```python
# Two features
X = np.array([[1, 2], [2, 3], [3, 4]])

# Polynomial features with interactions
poly_inter = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly_inter = poly_inter.fit_transform(X)
print("Polynomial with Interactions:\n", X_poly_inter)
print("Feature Names:", poly_inter.get_feature_names_out())
# Output:
# [[ 1.  2.  1.  2.  4.]
#  [ 2.  3.  4.  6.  9.]
#  [ 3.  4.  9. 12. 16.]]
# Feature Names: ['x0' 'x1' 'x0^2' 'x0 x1' 'x1^2']
```

## Predictions
```python
# Predict output using Linear Regression
linear_pred = lin_reg.predict([[330]])
print("Linear Regression Predicted Output:", linear_pred)

# Predict output using Polynomial Regression
poly_pred = model.predict(poly_reg.fit_transform([[330]]))
print("Polynomial Regression Predicted Output:", poly_pred)
```

- **Linear Regression predicted output:** `[330378.78787879]`
- **Polynomial Regression predicted output:** `[158862.45265155]`
- **Polynomial Regression provides a more accurate prediction** in cases where the dataset exhibits **non-linearity**.
- Choosing the right polynomial degree is crucial to avoid **overfitting** or **underfitting**.

---

## **4. Tools and Methods Summary**
- **Feature Generation**: `sklearn.preprocessing.PolynomialFeatures`.
- **Modeling**: `sklearn.linear_model.LinearRegression`.
- **Visualization**: `matplotlib.pyplot.plot()`, `scatter()`.
- **Evaluation**: `sklearn.metrics.r2_score` (below).

```python
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print("R² Score:", r2)  # Measures fit quality
```

```python
from sklearn.linear_model import Ridge

# Ridge regression with polynomial features
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_poly, y)
y_pred_ridge = model_ridge.predict(X_poly)
print("Ridge Coefficients:", model_ridge.coef_)
```

---
## **5. Key Considerations**
- **Degree Selection**: Higher degrees increase flexibility but risk overfitting.
- **Feature Scaling**: Polynomial terms amplify scale differences normalize first.
- **Computational Cost**: Number of features grows as \( \binom{n+d}{d} \) (where \(n\) = features, \(d\) = degree).
- **Regularization**: Use Ridge/Lasso with high-degree polynomials to control overfitting.
- **Optimal degree selection** is crucial for balancing bias and variance.

📌 **Key Takeaway:** Polynomial Regression is a **powerful extension of Linear Regression** that allows models to fit **non-linear data patterns** more effectively.

---

## Feature Interactions with Polynomial Features
Use `PolynomialFeatures` to include feature interactions in your model. However, this is impractical for large datasets and unnecessary for tree-based models.

### Example
```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 4, 4], 'C': [0, 10, 100]})

# Generate interaction-only features (exclude polynomial terms)
poly = PolynomialFeatures(include_bias=False, interaction_only=True)
poly.fit_transform(X)
```

---

## **4. Tools and Methods Summary**
- **Data**: `pandas.DataFrame()`, `sklearn.preprocessing.StandardScaler`.
- **Models**: `sklearn.linear_model.LogisticRegression`, `tensorflow.keras.models`.
- **Loss**: `np.mean()`, `sklearn.metrics.log_loss`.
- **Optimization**: Manual gradient descent, `tensorflow.keras.optimizers`.
- **Evaluation**: `sklearn.metrics.accuracy_score`, `r2_score`.
- **Regularization**: `penalty` in `sklearn` models.
- **Stats**: `scipy.stats`, `numpy.random`.

---
