# Stage 10: Modular Feature Engineering Script

## Overview
Feature engineering is a crucial step in machine learning that enhances model performance by transforming raw data into meaningful features. This guide presents an **Object-Oriented Approach (OOP)** to modular feature engineering, making the process reusable, structured, and scalable.

## Step 1: Object-Oriented Feature Engineering

The following **FeatureEngineering** class encapsulates various feature engineering techniques such as handling missing data, encoding categorical variables, scaling numerical features, creating new features, reducing dimensionality, and selecting important features.

### Implementation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

class FeatureEngineering:
    def __init__(self, data):
        self.data = data.copy()

    def handle_missing_data(self, strategy='mean'):
        """Handle missing data in the dataset."""
        imputer = SimpleImputer(strategy=strategy)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        return self

    def encode_categorical_features(self):
        """Encode categorical features in the dataset."""
        for col in self.data.select_dtypes(include='object').columns:
            encoder = LabelEncoder()
            self.data[col] = encoder.fit_transform(self.data[col])
        return self

    def scale_features(self):
        """Scale numerical features to standardize the data."""
        scaler = StandardScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        return self

    def create_new_features(self):
        """Create new features based on domain knowledge."""
        if 'alcohol' in self.data.columns and 'density' in self.data.columns:
            self.data['alcohol_density_ratio'] = self.data['alcohol'] / self.data['density']
        return self

    def reduce_dimensionality(self, n_components=5):
        """Reduce dimensionality of the dataset using PCA."""
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(self.data)
        self.data = pd.DataFrame(reduced_features, columns=[f'PC{i+1}' for i in range(n_components)])
        return self

    def select_important_features(self, target, k=5):
        """Select top-k features based on relevance to the target."""
        selector = SelectKBest(score_func=f_regression, k=k)
        selected_features = selector.fit_transform(self.data, target)
        self.data = pd.DataFrame(selected_features, columns=[f'Feature_{i+1}' for i in range(k)])
        return self

    def get_processed_data(self):
        """Return the processed dataset."""
        return self.data
```

## Step 2: Using the Feature Engineering Class

The following script demonstrates how to apply the **FeatureEngineering** class on a dataset.

```python
# Load the dataset
data = pd.read_csv('winequality.csv')

# Split features and target
X = data.drop('quality', axis=1)  # Features
y = data['quality']              # Target

# Initialize Feature Engineering class
fe = FeatureEngineering(X)

# Process data
processed_data = (fe.handle_missing_data(strategy='mean')
                    .encode_categorical_features()
                    .scale_features()
                    .create_new_features()
                    .reduce_dimensionality(n_components=5)
                    .get_processed_data())

# Display the processed data
print("Processed Data:\n", processed_data.head())
```

## Summary
This **modular feature engineering script** provides a reusable, scalable, and structured approach to feature transformation. By leveraging OOP, each method can be applied in a sequence, making the feature engineering process efficient and easy to manage.

### Key Benefits:
- **Modular & Reusable**: Can be applied to different datasets with minor modifications.
- **Scalable**: Easily extendable for more feature engineering techniques.
- **Structured & Readable**: Follows an object-oriented paradigm for clarity and maintainability.

This approach ensures that data preprocessing is efficient, improving model performance while maintaining a clean and organized workflow.

---
# Stage 10: Modular Feature Engineering Script

## Step 2: Functional Oriented Approach

### Step 1: Load the Dataset
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data successfully loaded. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None
```

### Step 2: Basic Dataset Information
```python
def basic_info(data):
    """Display basic information about the dataset."""
    print("\n--- Dataset Information ---")
    data.info()
    
    print("\n--- First Five Rows ---")
    print(data.head())
    
    print("\n--- Dataset Summary ---")
    print(data.describe(include='all'))
```

### Step 3: Missing Value Analysis
```python
def missing_values_analysis(data):
    """Analyze missing values in the dataset."""
    print("\n--- Missing Values ---")
    missing = data.isnull().sum()
    print(missing[missing > 0])
    
    # Visualize missing data
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()
```

### Step 4: Univariate Analysis
```python
def univariate_analysis(data, categorical_features, numerical_features):
    """Perform univariate analysis on categorical and numerical features."""
    for column in categorical_features:
        plt.figure()
        sns.countplot(x=column, data=data, palette="viridis")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.show()
    
    for column in numerical_features:
        plt.figure()
        sns.histplot(data[column], kde=True, bins=20, color="teal")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
```

### Step 5: Bivariate Analysis
```python
def bivariate_analysis(data, target_column, numerical_features):
    """Perform bivariate analysis between features and target."""
    for column in numerical_features:
        if column != target_column:
            plt.figure()
            sns.scatterplot(x=column, y=target_column, data=data, color="darkblue")
            plt.title(f"Relationship between {column} and {target_column}")
            plt.xlabel(column)
            plt.ylabel(target_column)
            plt.show()
```

### Step 6: Correlation Analysis
```python
def correlation_analysis(data):
    """Analyze correlations between numerical features."""
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title("Correlation Matrix")
    plt.show()
```

### Step 7: Outlier Detection
```python
def outlier_detection(data, numerical_features):
    """Detect outliers in numerical features."""
    for column in numerical_features:
        plt.figure()
        sns.boxplot(x=data[column], color="coral")
        plt.title(f"Outliers in {column}")
        plt.xlabel(column)
        plt.show()
```

### Step 8: Feature Engineering Suggestions
```python
def feature_engineering_suggestions():
    """Suggest feature engineering opportunities based on dataset analysis."""
    print("\n--- Feature Engineering Suggestions ---")
    print("1. Encoding categorical variables using techniques like one-hot encoding or label encoding.")
    print("2. Generating interaction features for highly correlated variables.")
    print("3. Scaling numerical features using standardization or normalization.")
    print("4. Creating bins for continuous variables (e.g., age groups).")
    print("5. Imputing missing values with mean, median, or predictive models.")
```

## Main Execution
```python
if __name__ == "__main__":
    # Load data
    file_path = "data/raw/Mall_Customers.csv"
    data = load_data(file_path)
    
    if data is not None:
        # Define feature types
        categorical_features = ['Gender', 'Customer_Category']  # Update based on dataset
        numerical_features = ['Age', 'Annual_Income', 'Spending_Score']  # Update based on dataset
        target_column = 'Spending_Score'  # Update based on dataset
        
        # Perform EDA steps
        basic_info(data)
        missing_values_analysis(data)
        univariate_analysis(data, categorical_features, numerical_features)
        bivariate_analysis(data, target_column, numerical_features)
        correlation_analysis(data)
        outlier_detection(data, numerical_features)
        feature_engineering_suggestions()
    
    print("Exploratory Data Analysis Complete.")
