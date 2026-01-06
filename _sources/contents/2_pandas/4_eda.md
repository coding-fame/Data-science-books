
# **What is Exploratory Data Analysis (EDA)?**
EDA is the process of analyzing and summarizing a dataset to uncover its structure, patterns, anomalies, and relationships. It involves statistical techniques and visualizations to gain insights, identify issues (e.g., missing data, outliers), and inform subsequent steps like feature engineering or modeling.

## **Goals of EDA**
- Understand data distribution and variability.
- Detect anomalies, outliers, and errors.
- Explore relationships between variables.
- Form hypotheses for further analysis.

## **Key Steps**
1. **Data Overview**: Structure, types, and summary statistics.
2. **Univariate Analysis**: Examine individual variables.
3. **Bivariate/Multivariate Analysis**: Explore relationships between variables.
4. **Data Cleaning**: Handle missing values, outliers, etc.
5. **Visualization**: Use plots to reveal insights.

# **EDA Techniques and Methods**
Let’s break this down into actionable steps with examples.

## 🛠️ Stage 0: Import Libraries and Load Data

```python
import pandas as pd
import numpy as np

# Sample dataset (e.g., Iris dataset from sklearn)
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target
```

## **a. Data Overview**
Understand the dataset’s structure and basic properties.

### 🔍 Stage 1: Data Inspection

1. **Check Size**: Find out how many rows (samples) and columns (features) you have.
   ```python
   print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
   ```

2. **List Columns**: See all column names to spot any issues.
   ```python
   print("Columns:", df.columns.tolist())
   ```

3. **Check Data Types**: Confirm each column’s type (e.g., numbers or text).
   ```python
   print("Data Types:\n", df.dtypes)
   ```

4. **Summarize Numbers**: Look at ranges for numerical columns.
   ```python
   print("Numerical Summary:\n", df.describe().T[['min', 'max', 'mean']])
   ```

## 🔢 Stage 2: Value Inspection

Look at the data inside your columns to check its quality.

### Steps
1. **Count Unique Values**: See how many different values each column has.
   ```python
   print("Unique Values:\n", df.nunique())
   ```

2. **Find Constant Columns**: Identify columns with just one value.
   ```python
   constant_columns = [col for col in df.columns if df[col].nunique() == 1]
   print("Constant Columns:", constant_columns)
   ```

3. **Check Category Frequencies**: For text columns, see how often each category appears.
   ```python
   for col in df.select_dtypes(include=['object']).columns:
       print(f"{col} Counts:\n", df[col].value_counts())
   ```

5. **Visualize Data**:
   - **Text Columns**: Bar plots show category counts.
   - **Number Columns**: Histograms show distributions.
   ```python
   for col in df.select_dtypes(include=['object']).columns:
       sns.countplot(data=df, x=col)
       plt.title(f"{col} Bar Plot")
       plt.show()
   for col in df.select_dtypes(include=['int64', 'float64']).columns:
       sns.histplot(data=df, x=col, bins=20, kde=True)
       plt.title(f"{col} Histogram")
       plt.show()
   ```

### Why It’s Important for ML
- **Unique Values**: Too many unique values in a text column might need special handling (e.g., grouping rare categories).
- **Constant Columns**: These don’t help ML models learn and can be removed.
- **Frequencies**: Uneven category counts (imbalance) might need fixing with techniques like oversampling.
- **Ranges**: Big differences in numbers (outliers) can confuse some ML models, like linear regression.

---

## 🎯 Stage 3: Target Variable Analysis

### What You’ll Do
Study the column you want to predict (the target).

### Steps
1. **Check Its Type**: Is it categories (e.g., “good/bad”) or numbers (e.g., a score)?
   ```python
   target = 'quality'  # Your target column
   print(f"Target Type: {df[target].dtype}")
   ```

2. **Look at Distribution**:
   - **Categories**: Count each class.
   - **Numbers**: See how values spread out.
   ```python
   if df[target].dtype == 'object' or df[target].nunique() < 25:
       sns.countplot(data=df, x=target)
       plt.title(f"{target} Class Counts")
   else:
       sns.histplot(df[target], kde=True, bins=20)
       plt.title(f"{target} Spread")
   plt.show()
   ```

3. **Spot Imbalance**: For categories, check if one dominates.
   ```python
   if df[target].dtype == 'object' or df[target].nunique() < 25:
       print("Class Percentages:\n", df[target].value_counts(normalize=True) * 100)
   ```

4. **Link to Other Columns**:
   - **Numbers**: Use boxplots to compare with the target.
   - **Text**: Use count plots to see patterns.
   ```python
   for col in df.select_dtypes(include=['int64', 'float64']).columns:
       if col != target:
           sns.boxplot(data=df, x=target, y=col)
           plt.title(f"{col} by {target}")
           plt.show()
   for col in df.select_dtypes(include=['object']).columns:
       if col != target:
           sns.countplot(data=df, x=col, hue=target)
           plt.title(f"{col} by {target}")
           plt.show()
   ```

5. **Check Correlations** (if target is a number):
   ```python
   if df[target].dtype != 'object':
       correlation = df.corr()[target].sort_values(ascending=False)
       print(f"Correlations with {target}:\n", correlation)
   ```

6. **Find Outliers**:
   ```python
   sns.boxplot(data=df, y=target)
   plt.title(f"{target} Outliers")
   plt.show()
   ```

### Why It’s Important for ML
- **Type**: A categorical target means classification (e.g., predicting “good” or “bad”), while a numerical target means regression (e.g., predicting a score).
- **Distribution**: If one category dominates, your model might ignore the others unless you balance it.
- **Links**: Features tied to the target are your best predictors—focus on them.
- **Outliers**: These can throw off predictions, especially in regression.

---

## 🧹 Stage 5: Data Cleaning & Transformation

Data cleaning fixes problems like missing values, duplicates, and outliers to make your data ready for ML models. Clean data helps models learn patterns accurately.

What You’ll Do
- Fill in or remove missing data.  
- Get rid of duplicates and handle extreme values.

### 1. Handling Missing Values

Missing values happen when data is incomplete or entered incorrectly. They can confuse ML models, so we need to fix them.

**Ways to Fix Missing Values**
- **Fill with a default value**: Use the mean (average), median (middle value), or mode (most common value) of a column.
- **Drop rows**: Remove rows with missing values if there’s not much missing data.
- **Impute values**: Use ML techniques (like predicting missing values) for better accuracy.
- Fill gaps with a number (like the mean) or a word (like "Unknown").  
- Drop rows if too much is missing.  

Example: Fill with Specific Value
```python
# Fill missing 'sales' values with 0
df['sales'].fillna(0, inplace=True)
```

Example: Drop Rows
```python
# Remove rows with missing values
df_cleaned = df.dropna()
```

   ```python
   df['numerical_col'] = df['numerical_col'].fillna(df['numerical_col'].mean())  # Fill with mean
   df['categorical_col'] = df['categorical_col'].fillna("Unknown")  # Fill with "Unknown"
   df = df.dropna()  # Remove rows with missing values
   ```
  - See where data is missing with a table or heatmap.  
  ```python
  missing_data = df.isnull().sum()
  print("Missing Values:\n", missing_data)
  sns.heatmap(df.isnull(), cbar=False, cmap='viridis')  # Yellow shows missing spots
  plt.title("Missing Values Heatmap")
  plt.show()
  ```

### 2. Handling Duplicates

Duplicate rows repeat the same information, which can trick ML models into overemphasizing certain data. Removing them keeps the dataset clean.

**Remove Duplicates**  
   - Delete repeated rows to avoid overrepresenting some data.  
   ```python
   df = df.drop_duplicates()
   ```

### 3. Handling Outliers

Outliers are unusual values that don’t fit the normal pattern. They can confuse ML models like regression or clustering, so we detect and manage them.

**Detect Outliers**
- **Boxplots**: they show extreme values as dots.
- **IQR Method**: Finds outliers using the Interquartile Range (IQR).
- **Z-Score**: Marks values far from the average (e.g., beyond 3 standard deviations).

   ```python
   for col in numerical_features:
       Q1 = df[col].quantile(0.25)  # 25th percentile
       Q3 = df[col].quantile(0.75)  # 75th percentile
       IQR = Q3 - Q1
       lower = Q1 - 1.5 * IQR
       upper = Q3 + 1.5 * IQR
       df[col] = df[col].clip(lower, upper)  # Cap values outside this range
   ```

```python
from scipy import stats

# Z-score for outlier detection
z_scores = np.abs(stats.zscore(df["sepal width (cm)"]))
outliers = df["sepal width (cm)"][z_scores > 3]
print("Outliers in Sepal Width:", outliers)
# Output: e.g., 4.4, 4.1 (if present)

# IQR method
Q1 = df["sepal width (cm)"].quantile(0.25)
Q3 = df["sepal width (cm)"].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df["sepal width (cm)"][(df["sepal width (cm)"] < Q1 - 1.5 * IQR) | 
                                      (df["sepal width (cm)"] > Q3 + 1.5 * IQR)]
print("Outliers via IQR:", outliers_iqr)
```

#### Fix Outliers
- **Remove**: Drop rows with outliers.
- **Cap**: Set extreme values to a maximum or minimum limit.
- **Transform**: Use math (e.g., log) to reduce their impact.

#### Example: Cap Outliers
```python
# Cap 'price' at lower and upper bounds
df['price'] = df['price'].clip(lower=lower_bound, upper=upper_bound)
```

### 2. Analyze Skewness
- Check if numerical data is lopsided (skewed). Fix it with transformations if needed.  
  ```python
  from scipy.stats import skew
  for col in numerical_features:
      print(f"{col} Skewness: {skew(df[col].dropna())}")  # Positive = right skew, Negative = left skew
  ```

### Why It Matters for ML
- **Missing Values**: Knowing what’s missing helps you decide whether to fill it (impute) or ignore it, keeping your dataset useful.  
- **Duplicates**: They trick the model into thinking some patterns are more common than they are.  
- **Outliers**: Extreme values can throw off models, especially ones sensitive to distances (like k-nearest neighbors).
- **Skewness**: Many ML models (like logistic regression) work best with balanced (normal) data. A log transform can fix heavy skewness.

---

# **c. Univariate Analysis**

Univariate analysis means looking at one feature (column) at a time to understand its properties. This helps you spot issues that could affect your ML model, like uneven data or extreme values.

Examine the distribution and characteristics of individual variables.

What You’ll Do
- Check each feature individually.
- Learn how its values are distributed (spread out).

Steps
1. **Separate Features by Type**  
   - **Numerical**: Numbers like "age" or "price."  
   - **Categorical**: Categories like "color" or "type."  
   Here’s how to identify them using Python:  
   ```python
   numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
   categorical_features = df.select_dtypes(include=['object']).columns.tolist()
   print("Numerical Features:", numerical_features)
   print("Categorical Features:", categorical_features)
   ```

2. **Analyze Numerical Features**  
   - Look at how values are spread with histograms.  
   - Get stats like average (mean) and middle value (median).  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   
   for col in numerical_features:
       sns.histplot(df[col], kde=True)  # kde adds a smooth curve
       plt.title(f"{col} Distribution")
       plt.show()
   print(df[numerical_features].describe())  # Shows mean, min, max, etc.
   ```

3. **Analyze Categorical Features**  
   - Count how often each category appears with bar plots.  
   ```python
   for col in categorical_features:
       sns.countplot(x=df[col])
       plt.title(f"{col} Bar Plot")
       plt.show()
   ```

### Why It Matters for ML
- **Numerical Features**: If data is skewed (lopsided) or has outliers (extreme values), it might confuse models like linear regression. You may need to adjust it later.  
- **Categorical Features**: If one category is much more common (imbalanced), your model might focus too much on it and ignore others. This can lead to biased predictions.

---

## 4. Handling Categorical Variables
If categorical variables exist, analyze their distribution.

ML models need numbers, not text. Categorical variables (like "red," "blue") must be turned into numbers.

### Methods
- **Label Encoding**: Assigns numbers to categories (e.g., "red" = 0, "blue" = 1). Good for ordered data.
- **One-Hot Encoding**: Creates new columns for each category (e.g., "is_red," "is_blue"). Best for unordered data.

#### Example: Label Encoding
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['color'] = encoder.fit_transform(df['color'])
```

#### Example: One-Hot Encoding
```python
df = pd.get_dummies(df, columns=['color'])
```

```python
# Convert target to categorical names
df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# Count plot
plt.figure(figsize=(8, 5))
sns.countplot(x="species", data=df, palette="pastel")
plt.title("Species Distribution")
plt.show()
# Insight: Balanced dataset (50 each)
```

---

## 5. Feature Scaling

Features (columns) with different scales (e.g., "age" in 20-80, "salary" in 0-100,000) can confuse ML models like K-Nearest Neighbors or Neural Networks. Scaling puts them on the same range.

### Methods
- **Standardization**: Adjusts values to have a mean of 0 and standard deviation of 1.
- **Min-Max Scaling**: Adjusts values to a range (e.g., 0 to 1).

#### Example: Standardization
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
```

#### Example: Min-Max Scaling
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
```

---

## 6. Handling Date/Time Data

Date/time data (like "2023-05-15") needs to be split into useful parts (e.g., year, month) for ML models.

#### Example: Convert and Extract
```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract year, month, day
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
```

---

## Bivariate Analysis (Two Variables)
Explore relationships between pairs of variables.

### Numerical vs. Numerical
- **Scatter Plot**: Shows how two numbers relate.
- **Correlation**: Measures strength and direction (e.g., 0.8 means strong positive link).

```python
# Scatter plot: Petal Length vs Petal Width
plt.figure(figsize=(8, 5))
sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="target", data=df, palette="deep")
plt.title("Petal Length vs Petal Width by Species")
plt.show()
# Insight: Clear clusters by target (species)

# Correlation matrix
corr = df.corr()
print("Correlation Matrix:")
print(corr)

# Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()
# Insight: High correlation between petal length and width
```

### Numerical vs. Categorical
- **Boxplot**: Compares a number across categories.

#### Example: Boxplot
```python
sns.boxplot(x='color', y='price', data=df)
plt.title("Price by Color")
plt.show()
```

---

## Multivariate Analysis (Multiple Variables)
Analyze interactions across multiple variables.

**Pair Plots**
Shows relationships between many numerical variables at once.

```python
# Pair plot
sns.pairplot(df, hue="target", palette="husl")
plt.show()
# Insight: Visualizes all pairwise relationships and distributions

# Grouped box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x="target", y="petal length (cm)", data=df, palette="Set2")
plt.title("Petal Length Distribution by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()
# Insight: Species differ significantly in petal length
```

### Correlation Heatmap
Checks how all numerical features relate.

#### Example
```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```

### Multicollinearity
When features are too similar, it confuses ML models. We use Variance Inflation Factor (VIF) to check this.

#### Example: VIF
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[['age', 'salary', 'price']]
vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
```

---

### **4. Tools and Methods Summary**
- **Data Overview**: `df.shape`, `df.info()`, `df.describe()`, `df.head()`.
- **Missing Data**: `df.isna()`, `sns.heatmap()`, `df.fillna()`.
- **Univariate**: `sns.histplot()`, `sns.boxplot()`, `df.describe()`.
- **Bivariate**: `sns.scatterplot()`, `sns.lineplot()`, `df.corr()`, `sns.heatmap()`.
- **Multivariate**: `sns.pairplot()`, `sns.catplot()`.
- **Outliers**: `stats.zscore()`, IQR method.
- **Categorical**: `sns.countplot()`, `df.value_counts()`.

---

# Time-Series Analysis (If Applicable)

Time-series analysis is useful when your data involves time, like daily sales or monthly temperatures. It helps you spot patterns such as trends (long-term changes), seasonality (repeating cycles), and unexpected changes (anomalies). These insights are vital for building ML models that predict future values.

## Why Use Time-Series Analysis?
- **Understand Patterns**: See how data changes over time.
- **Improve Predictions**: Help ML models forecast accurately.
- **Detect Issues**: Find unusual data points that might affect results.

---

## 1. Breaking Down the Time Series
Time-series decomposition splits the data into three parts:
- **Trend**: The overall direction (e.g., sales growing over years).
- **Seasonality**: Regular, repeating patterns (e.g., higher sales every December).
- **Residuals**: Random leftovers after removing trend and seasonality.

### Example: Decomposition in Python
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data and set 'Date' as the index
df = pd.read_csv('your_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Decompose with a yearly cycle (12 months)
decomposition = seasonal_decompose(df['Value'], model='additive', period=12)
decomposition.plot()
plt.show()
```

This code shows the trend, seasonality, and residuals in separate plots.

---

## 2. Checking Stationarity
A time series is **stationary** if its average and variation don’t change over time. Many ML models, like ARIMA, work better with stationary data. If it’s not stationary, you can adjust it (e.g., by differencing).

### Test for Stationarity: ADF Test
```python
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(df['Value'])
print(f"p-value: {adf_result[1]}")
if adf_result[1] < 0.05:
    print("Data is stationary")
else:
    print("Data is not stationary")
```

A p-value below 0.05 means the data is stationary.

---

## 3. Spotting Trends and Seasonality
A **rolling mean** smooths data to show trends over time.

### Example: Rolling Mean
```python
rolling_mean = df['Value'].rolling(window=12).mean()
plt.plot(df['Value'], label='Original Data')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.legend()
plt.show()
```

This plot highlights the long-term trend.

---

## 4. Exploring Relationships Over Time
**Autocorrelation** shows how past values relate to future ones, which is useful for forecasting.

### Example: Autocorrelation Plot
```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['Value'], lags=50)
plt.show()
```

Spikes in the plot indicate strong relationships at specific time lags.

---

## 5. Adding Time-Based Features
You can create features like “month” or “day of the week” to help ML models learn time patterns.

### Example: Date Features
```python
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
print(df.head())
```

---

# Feature Engineering

Feature engineering means creating or adjusting features to make ML models work better. Good features highlight patterns the model can learn.

---

## 1. Creating New Features
New features can come from math, categories, or dates.

### 1.1 Math-Based Features
Combine existing features (e.g., ratios).

#### Example: Sulfur Ratio
```python
df['sulfur_ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
```

### 1.2 Binning
Turn numbers into groups (e.g., age ranges).

#### Example: Age Groups
```python
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
```

### 1.3 Date Features
Extract parts of dates (e.g., month).

#### Example: Extract Month
```python
df['month'] = pd.to_datetime(df['date']).dt.month
```

---

## 2. Encoding Categories
ML models need numbers, not text. Encoding converts categories into numbers.

### 2.1 One-Hot Encoding
Make a column for each category (e.g., “red” becomes “is_red”).

#### Example
```python
df = pd.get_dummies(df, columns=['color'])
```

### 2.2 Label Encoding
Give each category a number (e.g., “red” = 0, “blue” = 1).

#### Example
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['color'] = encoder.fit_transform(df['color'])
```

---

## 3. Handling Outliers
Outliers are extreme values that can confuse models. You can limit them.

### Example: Cap Outliers
```python
low, high = df['price'].quantile([0.01, 0.99])
df['price'] = df['price'].clip(lower=low, upper=high)
```

---

## 4. Transforming Features
Transformations adjust feature scales or shapes for ML models.

### 4.1 Scaling
- **Min-Max Scaling**: Puts values between 0 and 1.
- **Standardization**: Centers data around 0 with a standard spread.

#### Min-Max Scaling

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['normalized_column'] = scaler.fit_transform(df[['column_name']])
```

#### Example: Standardization (Z-score Scaling)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
```

### 4.2 Polynomial Features
Combine features (e.g., age × salary) to capture relationships.

#### Example
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
df_poly = poly.fit_transform(df[['age', 'salary']])
```

---

## 5. Reducing Features
Too many features can slow models or cause overfitting. You can trim them.

### 5.1 Feature Selection
Keep only important features.

#### Example: Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
selector = RFE(model, n_features_to_select=5)
X_selected = selector.fit_transform(X, y)
```

### 5.2 Feature Extraction
Combine features into fewer ones with **PCA**.

#### Example: PCA
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

---

# Business Context Integration

This step connects data insights to business goals.

## Key Steps
- **Talk to Experts**: Confirm findings with people who know the business.
- **Explain Insights**: Describe patterns in a way that makes sense for the business.
- **Suggest Actions**: Recommend features or models to solve problems.

For example, if sales spike in December, you might suggest a feature for “holiday season” to improve predictions.

---

# Data Partitioning

Data partitioning splits your data into parts for training, validating, and testing ML models. This ensures the model works well on new data.

---

## 1. Random Split
Divide data randomly (e.g., 70% train, 30% test).

### Example
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
```

---

## 2. K-Fold Cross Validation
Split data into `k` parts, train on `k-1`, and test on 1. Repeat `k` times.

### Example
```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
```

This reduces overfitting.

---

## 3. Stratified Split
For imbalanced data (e.g., rare events), ensure each part reflects the full dataset.

### Example
```python
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
```

---

## 4. Balancing Imbalanced Data
If one class is rare, use **SMOTE** to create synthetic examples.

### Example
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

---

## Summary Table
| Method              | Use Case                 | Benefit                     |
|---------------------|--------------------------|-----------------------------|
| Random Split        | General split            | Simple and fair             |
| K-Fold              | Model evaluation         | Reduces overfitting         |
| Stratified Split    | Imbalanced data          | Keeps class balance         |
| SMOTE               | Rare classes             | Balances data synthetically |


