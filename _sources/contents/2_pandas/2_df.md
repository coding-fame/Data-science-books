# Pandas DataFrame - Complete Reference Guide

## 1. DataFrame Fundamentals

### What is a DataFrame?
A **DataFrame** is a 2-dimensional labeled data structure in pandas, similar to a spreadsheet or SQL table. It consists of rows (observations) and columns (features/variables), making it ideal for structured data analysis in machine learning workflows.

**Key Characteristics:**
- **Rows**: Indexed (default: 0-based integers or custom labels)
- **Columns**: Named (each column is a `Series` object)
- **Data**: Heterogeneous (supports mixed data types)
- **Memory Efficient**: Optimized for large datasets

### Creating DataFrames
```python
import pandas as pd
import numpy as np

# From dictionary (most common)
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# From lists
df = pd.DataFrame([['Alice', 25], ['Bob', 30]], columns=['Name', 'Age'])

# From NumPy array
array = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(array, columns=['A', 'B'])

# Empty DataFrame
df_empty = pd.DataFrame()
```

---

## 2. Essential DataFrame Attributes

| Attribute | Description | ML Relevance |
|-----------|-------------|--------------|
| `.shape` | (rows, columns) tuple | Dataset size for train/test splits |
| `.columns` | Column names Index | Feature names for model training |
| `.dtypes` | Data types per column | Type checking for ML algorithms |
| `.index` | Row index details | Time series indexing |
| `.values` | NumPy array of data | Direct input to ML models |
| `.size` | Total elements (rows × columns) | Memory optimization |
| `.empty` | Boolean for empty check | Data validation |

```python
# Practical usage in ML workflow
print(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Feature types: {df.dtypes}")
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
```

---

## 3. Core DataFrame Methods

### Data Exploration Methods
```python
# Basic inspection
df.head(3)        # First n rows
df.tail(2)        # Last n rows  
df.info()         # Structure overview
df.describe()     # Statistical summary

# Data quality
df.isnull().sum() # Missing values count
df.nunique()      # Unique values per column
df.duplicated().sum() # Duplicate rows
```

### Data Type Management
```python
# Type conversion strategies
df['age'] = df['age'].astype('int32')  # Memory efficiency
df['category'] = df['category'].astype('category')  # Categorical encoding

# Bulk conversion
type_map = {'age': 'float32', 'salary': 'float64'}
df = df.astype(type_map)
```

---

## 4. Data Cleaning & Preprocessing

### Handling Missing Values
```python
# Detection
missing_summary = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Treatment strategies
# 1. Removal
df_clean = df.dropna()  # Remove rows with any NaN
df_clean = df.dropna(subset=['age', 'income'])  # Specific columns

# 2. Imputation
df_filled = df.fillna({
    'age': df['age'].median(),    # Numerical - robust to outliers
    'category': 'Unknown',        # Categorical - new category
    'income': df['income'].mean() # Numerical - normal distribution
})

# 3. Advanced imputation (scikit-learn)
from sklearn.impute import SimpleImputer, KNNImputer
imputer = SimpleImputer(strategy='median')
df[['age']] = imputer.fit_transform(df[['age']])
```

### Outlier Management
```python
# IQR Method for outlier detection
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

# Cap outliers
lower, upper = detect_outliers_iqr(df['price'])
df['price'] = df['price'].clip(lower, upper)
```

---

## 5. Data Selection & Filtering

### Label-based Selection (`loc`)
```python
# Single row/column
df.loc[0]                    # Row by label
df.loc[:, 'age']             # Column by name

# Multiple selection
df.loc[0:5, ['name', 'age']] # Row range and specific columns
df.loc[df['age'] > 30, :]    # Boolean indexing

# Complex conditions
mask = (df['age'] > 25) & (df['income'] > 50000)
df_filtered = df.loc[mask, ['name', 'city']]
```

### Position-based Selection (`iloc`)
```python
df.iloc[0]           # First row
df.iloc[:, 0]        # First column  
df.iloc[0:5, 1:3]    # Row and column slices
df.iloc[[0, 2, 4]]   # Specific rows by position
```

### Boolean Filtering
```python
# Single condition
high_income = df[df['income'] > 100000]

# Multiple conditions
premium_customers = df[
    (df['income'] > 100000) & 
    (df['age'].between(30, 50)) &
    (df['city'].isin(['NYC', 'SF']))
]

# String operations
gmail_users = df[df['email'].str.contains('gmail.com', na=False)]
```

---

## 6. Data Transformation

### Column Operations
```python
# Renaming
df.rename(columns={'old_name': 'new_name'}, inplace=True)
df.columns = [col.lower() for col in df.columns]  # Standardize case

# Adding/removing columns
df['full_name'] = df['first_name'] + ' ' + df['last_name']
df = df.drop(columns=['temp_column'])

# Mathematical operations
df['bmi'] = df['weight'] / (df['height'] ** 2)
df['log_income'] = np.log(df['income'])
```

### Sorting & Ordering
```python
# Single column sort
df_sorted = df.sort_values('salary', ascending=False)

# Multi-column sort
df_sorted = df.sort_values(['department', 'salary'], 
                          ascending=[True, False])

# Index sorting
df_sorted = df.sort_index(ascending=False)
```

---

## 7. Aggregation & Grouping

### Basic Aggregations
```python
# Single column statistics
df['age'].mean()      # Average
df['salary'].median() # Median
df['category'].mode() # Most frequent
df['income'].std()    # Standard deviation
```

### GroupBy Operations
```python
# Single grouping
department_stats = df.groupby('department')['salary'].agg(['mean', 'std', 'count'])

# Multiple groupings
regional_sales = df.groupby(['region', 'product'])['sales'].sum()

# Custom aggregations
def salary_range(series):
    return series.max() - series.min()

summary = df.groupby('department').agg({
    'salary': ['mean', salary_range],
    'age': 'median',
    'employee_id': 'count'
})
```

### Advanced: Named Aggregations
```python
result = df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    total_employees=('employee_id', 'count'),
    salary_spread=('salary', lambda x: x.max() - x.min())
)
```

---

## 8. Data Merging & Combining

### SQL-style Joins
```python
# Inner join (default)
merged = pd.merge(customers, orders, on='customer_id')

# Left join
merged = pd.merge(customers, orders, on='customer_id', how='left')

# Multiple key join
merged = pd.merge(df1, df2, on=['date', 'product_id'])

# Indicator for join analysis
merged = pd.merge(df1, df2, how='outer', indicator=True)
```

### Concatenation
```python
# Vertical stacking
combined = pd.concat([df1, df2, df3], ignore_index=True)

# Horizontal combining
wide_data = pd.concat([df1, df2], axis=1)

# Real-world: Multiple CSV files
import glob
csv_files = glob.glob("data/*.csv")
dataframes = [pd.read_csv(file) for file in csv_files]
combined_data = pd.concat(dataframes, ignore_index=True)
```

---

## 9. Function Application

### Element-wise Operations
```python
# Series operations
df['name_upper'] = df['name'].str.upper()
df['email_domain'] = df['email'].str.split('@').str[1]

# Using apply for complex transformations
def age_group(age):
    if age < 30: return 'Young'
    elif age < 50: return 'Middle'
    else: return 'Senior'

df['age_group'] = df['age'].apply(age_group)

# Lambda functions
df['income_category'] = df['income'].apply(
    lambda x: 'High' if x > 100000 else 'Low'
)
```

### Row-wise Operations
```python
def calculate_bonus(row):
    base = row['salary'] * 0.1
    if row['performance'] == 'Excellent':
        return base * 1.5
    return base

df['bonus'] = df.apply(calculate_bonus, axis=1)
```

---

## 10. DateTime Operations

### Date Conversion & Extraction
```python
# String to datetime
df['date'] = pd.to_datetime(df['date_string'])
df['date'] = pd.to_datetime(df['date_string'], format='%Y-%m-%d')

# Component extraction
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.day_name()

# Time-based filtering
recent_data = df[df['date'] > '2023-01-01']
q1_data = df[df['date'].between('2023-01-01', '2023-03-31')]
```

### Time Series Operations
```python
# Set datetime index
df_ts = df.set_index('date')

# Resampling
daily_sales = df_ts['sales'].resample('D').sum()
monthly_avg = df_ts['price'].resample('M').mean()

# Rolling statistics
df_ts['7day_avg'] = df_ts['sales'].rolling(window=7).mean()
```

---

## 11. Advanced ML Preparation

### Feature Engineering
```python
# Binning continuous variables
df['age_group'] = pd.cut(df['age'], 
                        bins=[0, 18, 35, 60, 100],
                        labels=['Child', 'Young', 'Adult', 'Senior'])

# Creating interaction terms
df['income_age_interaction'] = df['income'] * df['age']

# Polynomial features
df['age_squared'] = df['age'] ** 2
```

### Encoding Categorical Variables
```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category', 'city'])

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Target encoding (for high cardinality)
target_mean = df.groupby('category')['target'].mean()
df['category_target_encoded'] = df['category'].map(target_mean)
```

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (zero mean, unit variance)
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# Normalization (0 to 1 range)
minmax = MinMaxScaler()
df[['age_norm', 'income_norm']] = minmax.fit_transform(df[['age', 'income']])
```

---

## 12. Performance Optimization

### Memory Efficiency
```python
# Downcasting numerical types
df['age'] = pd.to_numeric(df['age'], downcast='integer')
df['price'] = pd.to_numeric(df['price'], downcast='float')

# Using categorical types
df['category'] = df['category'].astype('category')

# Memory usage analysis
print(df.info(memory_usage='deep'))
```

### Efficient Operations
```python
# Use vectorized operations instead of apply
# Slow:
df['new_col'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# Fast:
df['new_col'] = df['a'] + df['b']

# Use query for complex filtering
result = df.query('age > 30 and income > 50000')
```

---

## 13. Data Export & Integration

### Saving Results
```python
# CSV (most common)
df.to_csv('processed_data.csv', index=False)

# Excel
df.to_excel('results.xlsx', sheet_name='Processed')

# Pickle (Python-only, preserves dtypes)
df.to_pickle('data.pkl')

# Database
import sqlite3
conn = sqlite3.connect('database.db')
df.to_sql('table_name', conn, if_exists='replace')
```

### Integration with ML Libraries
```python
# Convert to NumPy for scikit-learn
X = df[['age', 'income', 'education']].values
y = df['target'].values

# Convert to PyTorch tensors
import torch
X_tensor = torch.tensor(X, dtype=torch.float32)

# Convert to TensorFlow datasets
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices((X, y))
```

---

## 14. Best Practices Checklist

### ✅ Data Quality
- [ ] Check for missing values
- [ ] Identify and handle outliers
- [ ] Remove duplicate records
- [ ] Validate data types
- [ ] Check for data consistency

### ✅ Feature Engineering
- [ ] Create relevant derived features
- [ ] Handle categorical variables appropriately
- [ ] Scale numerical features if needed
- [ ] Address multicollinearity
- [ ] Create time-based features for temporal data

### ✅ ML Preparation
- [ ] Split data into train/validation/test sets
- [ ] Ensure no data leakage between splits
- [ ] Balance classes if needed
- [ ] Create feature and target sets
- [ ] Set up cross-validation strategy

### ✅ Performance
- [ ] Use appropriate data types
- [ ] Prefer vectorized operations
- [ ] Process data in chunks if memory-constrained
- [ ] Cache intermediate results
- [ ] Profile memory usage

