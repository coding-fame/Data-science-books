
# **What is Feature Engineering?**

Feature engineering is the process of creating, selecting, and transforming raw data into features (input variables) that enhance a machine learning model’s ability to learn patterns and make predictions. Well-engineered features can often outweigh the choice of algorithm in terms of impact on model performance.

## **Goals**
- Improve model accuracy and interpretability.
- Reduce noise and redundancy.
- Handle missing data, outliers, and domain-specific requirements.

## **Key Aspects**
- **Creation**: Generating new features from existing data.
- **Transformation**: Scaling, encoding, or normalizing data.
- **Selection**: Choosing the most relevant features.

---

## 🎯 Why This Matters?
- **Impact**: Proper preprocessing can improve model accuracy by 15-30%.

> 🧠 *Real-World Analogy:* "Just like sharpening tools before building a house - preprocessing shapes raw data for optimal model performance"

---

# **1. Common Feature Engineering Techniques**

## **a. Handling Missing Data**
Missing values can skew models; feature engineering addresses this by imputation or creating indicators.

Four Common Approaches:
1. **Drop rows** with NaNs.
2. **Drop columns** with NaNs.
3. **Fill NaNs** with imputed values.
4. **Use models that handle NaNs**.

```python
import pandas as pd
import numpy as np

# Sample DataFrame with missing values
data = pd.DataFrame({
    "Age": [25, np.nan, 30, 35],
    "Salary": [50000, 60000, np.nan, 75000]
})

# Imputation with mean
data["Age_filled"] = data["Age"].fillna(data["Age"].mean())
print(data)
# Output:
#    Age  Salary  Age_filled
# 0  25.0  50000        25.0
# 1   NaN  60000        30.0
# 2  30.0    NaN        30.0
# 3  35.0  75000        35.0

# Missing indicator
data["Salary_missing"] = data["Salary"].isna().astype(int)
print(data)
# Output:
#    Age  Salary  Age_filled  Salary_missing
# 0  25.0  50000        25.0               0
# 1   NaN  60000        30.0               0
# 2  30.0    NaN        30.0               1
# 3  35.0  75000        35.0               0
```

### 3. Missing Value Imputation
Missing values can disrupt model training. Here’s how to handle them:

- **Simple Imputation**: Use mean, median, or mode.
- **Advanced Imputation**: Use algorithms like KNN to estimate missing values.

#### 🧩 Code Example:
```python
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

data = np.array([[1, np.nan], [3, 4], [np.nan, 6]])

# Mean Imputation
imputer = SimpleImputer(strategy='mean')
imputed_data_mean = imputer.fit_transform(data)
print("Mean Imputed Data:\n", imputed_data_mean)

# KNN Imputation (Advanced)
knn_imputer = KNNImputer(n_neighbors=2)
imputed_data_knn = knn_imputer.fit_transform(data)
print("KNN Imputed Data:\n", imputed_data_knn)
```

**Why It Matters**: ML models like **Decision Trees** can handle missing values, but most require complete data.

---

### Imputing Categorical Missing Values
Use the most frequent category or a new category like "Unknown":
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
df['Category'] = imputer.fit_transform(df[['Category']])
```

---

## **c. Scaling and Normalization**
Features with different scales (e.g., age vs. salary) can bias models. Scaling ensures uniformity.

- **StandardScaler**: Zero mean, unit variance.
- **MinMaxScaler**: Scales to a range (e.g., 0 to 1).

### 🧪 What is Feature Scaling?
**Feature Scaling** is a technique used to standardize the independent features present in the data within a fixed range.

Feature Scaling standardizes the range of numerical features so that no single feature dominates the model due to its scale. This is especially important for algorithms like:
- **Gradient Descent** in ML.
- **Backpropagation** in Deep Learning (DL).
- **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**, which rely on distance calculations.

| **Technique**      | **Formula**                               | **When to Use**                              | **Code Example**                  |
|--------------------|-------------------------------------------|----------------------------------------------|-----------------------------------|
| **Min-Max Scaling**| \( `(x - min(x)) /(max(x) - min(x))` \) | When you need data in a fixed range (0-1).    | `MinMaxScaler(feature_range=(0,1))` |
| **Standard Scaling**| \( `(x - mu) /sigma` \)              | When data is normally distributed.           | `StandardScaler()`                |
| **Robust Scaling** | \( `(x - Q1) /(Q3 - Q1)` \)              | When data contains outliers.                 | `RobustScaler()`                  |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.DataFrame({"Age": [25, 30, 35], "Salary": [50000, 60000, 75000]})

# Standardization
scaler = StandardScaler()
data[["Age_std", "Salary_std"]] = scaler.fit_transform(data[["Age", "Salary"]])
print(data)
# Output:
#    Age  Salary   Age_std  Salary_std
# 0   25   50000 -1.161895   -1.161895
# 1   30   60000  0.000000    0.000000
# 2   35   75000  1.161895    1.161895

# Min-Max Scaling
minmax = MinMaxScaler()
data[["Age_minmax", "Salary_minmax"]] = minmax.fit_transform(data[["Age", "Salary"]])
print(data)
# Output includes:
#    Age_minmax  Salary_minmax
# 0         0.0            0.0
# 1         0.5            0.4
# 2         1.0            1.0
```

**Why It Matters**: Algorithms like KNN and SVM are sensitive to feature scales. Scaling ensures fair distance calculations.

---

## 2. Outlier Handling Strategies
Outliers are extreme values that can skew model performance. 

Ways to handle outliers:

- **Drop outliers**
- **Mark outliers using boolean conditions**
- **Transform outliers into features**

Here's how to handle them:

| **Method**      | **Use Case**                 | **Code Snippet**                                      |
|-----------------|------------------------------|-------------------------------------------------------|
| **IQR Filter**  | Moderate outliers            | ```python\nQ1 = df.quantile(0.25)\nQ3 = df.quantile(0.75)\nIQR = Q3 - Q1\nfiltered_df = df[(df >= Q1 - 1.5*IQR) & (df <= Q3 + 1.5*IQR)]\n``` |
| **Z-Score**     | Extreme values               | ```python\nfrom scipy import stats\nz = np.abs(stats.zscore(data))\nfiltered_data = data[z < 3]\n``` |
| **Capping**     | Preserve data shape          | ```python\ndf = df.clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)\n``` |

> **Pro Tip**: Visualize outliers using boxplots before deciding on a strategy.

**Why It Matters**: Outliers can mislead models like **Linear Regression**, causing poor predictions.

---

## **Encoding Categorical Variables**

Categorical data represents categories (e.g., colors, cities) and needs to be converted into numbers for ML models.

### Types of Categorical Data:
1. **Nominal**: No order (e.g., colors: red, blue, green).
2. **Ordinal**: Has order (e.g., ratings: low, medium, high).

In practice, nominal variables are more common.

Machine learning models require numerical inputs, so categorical data must be transformed.

- **Label Encoding**: Assigns integers to categories.
- **One-Hot Encoding**: Creates binary columns for each category.

---

### Label Encoding

Label encoding converts categorical labels into numerical values.

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample data
data = pd.DataFrame({"City": ["NY", "LA", "NY", "SF"]})

# Label Encoding
le = LabelEncoder()
data["City_Label"] = le.fit_transform(data["City"])
print(data)
# Output:
#   City  City_Label
# 0   NY          1
# 1   LA          0
# 2   NY          1
# 3   SF          2

# One-Hot Encoding
ohe = pd.get_dummies(data["City"], prefix="City")
data = pd.concat([data, ohe], axis=1)
print(data)
# Output:
#   City  City_Label  City_LA  City_NY  City_SF
# 0   NY          1        0        1        0
# 1   LA          0        1        0        0
# 2   NY          1        0        1        0
# 3   SF          2        0        0        1
```

### Ordinal Encoding

Ordinal encoding assigns integer values to each category, but be mindful that it may introduce unintended relationships.

```python
from sklearn.preprocessing import OrdinalEncoder

data = [['blue'], ['green'], ['red']]
encoder = OrdinalEncoder()
result = encoder.fit_transform(data)
```

> **Problem with Ordinal Encoding**: If applied to nominal data, it can introduce a false relationship between categories, which might lead to poor model performance. Use **One-Hot Encoding** to avoid this issue.

### One-Hot Encoding

One-Hot Encoding creates binary columns for each category, ensuring no implied relationship between categories.

```python
from sklearn.preprocessing import OneHotEncoder

a = [['apple'], ['pear'], ['apple'], ['pear'], ['apple']]
encoder = OneHotEncoder(sparse=False)
onehot = encoder.fit_transform(a)
```

### Dummy Variable Encoding

Similar to One-Hot Encoding but drops one column to avoid redundancy

Dummy variable encoding drops one category to avoid redundancy. It is typically used in situations where the first category is implied.

```python
encoder = OneHotEncoder(drop='first', sparse=False)
onehot = encoder.fit_transform(data)
```

Encoding converts categories into numbers:

| **Technique**      | **Best For**          | **Dimensionality** | **Code Example**               |
|--------------------|-----------------------|--------------------|--------------------------------|
| **One-Hot Encoding**| Nominal data          | High               | `OneHotEncoder()`              |
| **Label Encoding** | Ordinal data          | Low                | `LabelEncoder()`               |
| **Target Encoding**| High-cardinality data | Medium             | `category_encoders.TargetEncoder()` |

---

### 2. Advanced Techniques

#### 🪄 Handling Rare Categories
Group rare categories into "Other" to reduce dimensionality:
```python
top_categories = df['Category'].value_counts().nlargest(5).index
df['Category'] = np.where(df['Category'].isin(top_categories), df['Category'], 'Other')
```

#### 🎯 Target Encoding
Encode categories based on the target variable’s mean:
```python
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df['Encoded_Cat'] = encoder.fit_transform(df['Category'], df['Target'])
```

**Why It Matters**: Reduces dimensionality in high-cardinality data, improving model efficiency.

---

### **d. Feature Creation**
New features can be derived from existing ones (e.g., ratios, interactions).

```python
# Sample data
data = pd.DataFrame({
    "Height": [160, 170, 180],
    "Weight": [60, 70, 80]
})

# Creating BMI feature
data["BMI"] = data["Weight"] / (data["Height"] / 100) ** 2
print(data)
# Output:
#    Height  Weight        BMI
# 0     160      60  23.437500
# 1     170      70  24.221453
# 2     180      80  24.691358

# Interaction feature
data["Height_Weight"] = data["Height"] * data["Weight"]
print(data)
# Output includes:
#    Height_Weight
# 0          9600
# 1         11900
# 2         14400
```

---

### **e. Binning Continuous Variables**
Convert continuous data into discrete categories.

```python
# Binning Age
data = pd.DataFrame({"Age": [15, 25, 35, 45, 55]})
bins = [0, 18, 30, 50, 100]
labels = ["Teen", "Young", "Adult", "Senior"]
data["Age_Group"] = pd.cut(data["Age"], bins=bins, labels=labels)
print(data)
# Output:
#    Age Age_Group
# 0   15      Teen
# 1   25     Young
# 2   35     Adult
# 3   45     Adult
# 4   55    Senior
```

---

### **f. Handling Datetime Features**
Extract meaningful components from dates/times.

```python
# Sample datetime data
data = pd.DataFrame({"Date": pd.to_datetime(["2023-01-15", "2023-06-20", "2023-12-25"])})

# Extract features
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["Weekday"] = data["Date"].dt.weekday
print(data)
# Output:
#         Date  Year  Month  Day  Weekday
# 0 2023-01-15  2023      1   15        6
# 1 2023-06-20  2023      6   20        1
# 2 2023-12-25  2023     12   25        0
```

---

### **g. Text Feature Engineering**
Extract features from text data (e.g., word counts, TF-IDF).

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample text data
data = pd.Series(["I love coding", "Coding is fun", "I love Python"])

# Word count (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)
print(vectorizer.get_feature_names_out())  # Output: ['coding' 'fun' 'is' 'love' 'python']
print(X.toarray())
# Output:
# [[1 0 0 1 0]
#  [1 1 1 0 0]
#  [0 0 0 1 1]]

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data)
print(X_tfidf.toarray())
```

---

### **h. Polynomial Features**
Generate polynomial and interaction terms.

```python
from sklearn.preprocessing import PolynomialFeatures

data = pd.DataFrame({"X": [1, 2, 3], "Y": [4, 5, 6]})
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data)
print(poly.get_feature_names_out())  # Output: ['X' 'Y' 'X^2' 'X Y' 'Y^2']
print(poly_features)
# Output:
# [[ 1.  4.  1.  4. 16.]
#  [ 2.  5.  4. 10. 25.]
#  [ 3.  6.  9. 18. 36.]]
```

---

# **2. Feature Selection**
Not all features are useful—selecting the best reduces noise and computation.

## **a. Filter Methods**
- Use statistical measures (e.g., correlation).

```python
# Correlation-based selection
data = pd.DataFrame({
    "X1": [1, 2, 3, 4],
    "X2": [2, 4, 6, 8],
    "Target": [0, 1, 0, 1]
})
corr = data.corr()["Target"].abs()
print(corr)  # X2 is perfectly correlated with Target
# Output:
# X1        0.0
# X2        1.0
# Target    1.0
```

## **b. Wrapper Methods**
- Use a model to evaluate feature subsets (e.g., Recursive Feature Elimination).

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = data[["X1", "X2"]]
y = data["Target"]
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=1)
fit = rfe.fit(X, y)
print("Selected feature:", X.columns[fit.support_])  # Output: Index(['X2'], dtype='object')
```

## **c. Embedded Methods**
- Feature importance from models (e.g., tree-based).

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances)  # X2 is more important
# Output:
# X1    0.0
# X2    1.0
```

---

## **3. Practical Example: House Price Prediction**
Let’s tie it all together with a realistic dataset.

```python
# Sample house data
data = pd.DataFrame({
    "Size": [1500, 2000, 1800, np.nan],
    "Rooms": [3, 4, 3, 5],
    "Built": ["2010-05-01", "2000-12-15", "2015-03-10", "1995-08-20"],
    "Location": ["Urban", "Suburban", "Urban", "Rural"],
    "Price": [300000, 400000, 350000, 250000]
})

# Step 1: Handle missing data
data["Size"] = data["Size"].fillna(data["Size"].median())

# Step 2: Datetime features
data["Built"] = pd.to_datetime(data["Built"])
data["Age"] = (pd.Timestamp("2025-03-02") - data["Built"]).dt.days / 365

# Step 3: Encoding categorical variables
data = pd.get_dummies(data, columns=["Location"], prefix="Loc")

# Step 4: Feature creation
data["Size_per_Room"] = data["Size"] / data["Rooms"]

# Step 5: Scaling
scaler = StandardScaler()
data[["Size", "Age"]] = scaler.fit_transform(data[["Size", "Age"]])

print(data)
# Output (simplified):
#       Size  Rooms      Built       Age  Price  Loc_Rural  Loc_Suburban  Loc_Urban  Size_per_Room
# 0 -1.297771      3 2010-05-01 -0.137158  300000          0             0          1      500.000000
# 1  0.866514      4 2000-12-15  1.191053  400000          0             1          0      500.000000
# 2 -0.123743      3 2015-03-10 -0.701090  350000          0             0          1      600.000000
# 3  0.555000      5 1995-08-20  1.352747  250000          1             0          0      360.000000

# Step 6: Feature selection (correlation with Price)
corr = data.corr()["Price"].abs()
print(corr.sort_values(ascending=False))
# Output (example):
# Price            1.000000
# Size             0.986754
# Size_per_Room    0.933394
# Rooms            0.188982
# Age              0.086066
```

---

## **4. Tools and Libraries**
- **`pandas`**: Data manipulation (`fillna`, `get_dummies`, `cut`).
- **`numpy`**: Mathematical operations.
- **`scikit-learn`**: Preprocessing (`StandardScaler`, `PolynomialFeatures`), feature selection (`RFE`), and modeling.
- **`category_encoders`**: Advanced encoding (e.g., Target Encoding).
  ```python
  from category_encoders import TargetEncoder
  encoder = TargetEncoder()
  data["Location_encoded"] = encoder.fit_transform(data["Location"], data["Price"])
  ```
- **`featuretools`**: Automated feature engineering.
  ```python
  import featuretools as ft
  es = ft.EntitySet(id="house_data")
  es = es.add_dataframe(dataframe_name="houses", dataframe=data, index="index")
  features, feature_defs = ft.dfs(entityset=es, target_dataframe_name="houses")
  ```

---

## 🛠️ Level 3: Pipeline Power

### 1. The Assembly Line Approach
**Pipelines** automate preprocessing and modeling steps, ensuring consistency and reducing errors.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

numerical_cols = ['Age', 'Income']
categorical_cols = ['Gender', 'Occupation']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
```

**Success Metric**: Reduce deployment errors by 40% using pipelines.

**Why It Matters**: Pipelines ensure that preprocessing is applied consistently across training and testing data, avoiding data leakage.

---
## 🚀 Interview Prep Kit

### Common Questions & Answers
1. **Q:** "Why is feature scaling important?"  
   **A:** It ensures features with larger ranges don’t dominate the model, improving performance in algorithms like KNN or SVM.

2. **Q:** "When to use One-Hot vs. Label Encoding?"  
   **A:** Use **One-Hot Encoding** for nominal data and **Label Encoding** for ordinal data.

3. **Q:** "How to handle high-cardinality categorical data?"  
   **A:** Use **Target Encoding** or group rare categories into "Other."


