# Data Preprocessing Guide

---

## 1. Introduction to Preprocessing in Machine Learning

Preprocessing transforms raw data into a format suitable for ML models. It improves model performance, prevents errors, and ensures reliable predictions. Without preprocessing, models may overfit (memorize data instead of learning patterns) or produce biased results.

### Why Preprocessing Matters
- Ensures data is clean and consistent.
- Improves model accuracy and generalization to new data.
- Prevents issues like data leakage (when test data influences training).

### Common Preprocessing Tasks
- **Handling Missing Values**: Filling or removing incomplete data to avoid errors.
- **Encoding Categorical Variables**: Converting text categories (e.g., "male", "female") to numbers.
- **Scaling Features**: Adjusting numerical values to a common scale (e.g., between 0 and 1).
- **Feature Engineering**: Creating new features to help the model learn better.
- **Handling Outliers**: Reducing the impact of extreme values that can distort results.
- **Data Splitting**: Separating training and test data to prevent data leakage.

#### Example
If your dataset has missing values in the "Age" column, you might fill them with the median age. For a categorical column like "Gender", you encode it numerically (e.g., "male" = 0, "female" = 1) because ML models require numerical inputs.

---

## 2. What is `ColumnTransformer`?

`ColumnTransformer` applies different transformations to specific columns in your dataset. It is useful for datasets with mixed data types, such as numerical and categorical columns.

### Why Use `ColumnTransformer`?
- Saves time by handling multiple transformations in one step.
- Keeps your code organized and easy to follow.
- Works well with pipelines for a smooth ML workflow.

### Custom Feature Engineering with `FunctionTransformer`

`FunctionTransformer` lets you create custom transformations, such as mathematical operations or text extraction, and integrate them into preprocessing. This is useful for tasks like clipping values (limiting them to a range) or extracting parts of text data.

#### Example: Clipping Values and Extracting Letters

```python
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

# Sample dataset
X = pd.DataFrame({
    'Fare': [200, 300, 50, 900],           # Numerical column
    'Code': ['X12', 'Y20', 'Z7', np.nan],  # Text column with missing value
    'Deck': ['A101', 'C102', 'A200', 'C300']  # Text column
})

# Clip 'Fare' values between 100 and 600 to reduce outliers
clip_values = FunctionTransformer(np.clip, kw_args={'a_min': 100, 'a_max': 600})

# Extract first letter from 'Code' and 'Deck'
def first_letter(df):
    return df.apply(lambda x: x.str.slice(0, 1))

get_first_letter = FunctionTransformer(first_letter)

# Apply transformations to specific columns
ct = make_column_transformer(
    (clip_values, ['Fare']),               # Clip 'Fare'
    (get_first_letter, ['Code', 'Deck'])   # Extract first letter from 'Code' and 'Deck'
)

# Fit and transform the data
ct.fit_transform(X)
```

**Explanation**:
- **Clipping**: Limits "Fare" values to 100–600, reducing the impact of outliers (extreme values) on the model.
- **Text Extraction**: Pulls the first letter from "Code" and "Deck", creating new categorical features that may be useful for the model.

---

## 3. How to Select Columns in `ColumnTransformer`

You can select columns in seven ways for transformation:
- **By Name**: Use column names, e.g., `['Age', 'Sex']`.
- **By Position**: Use numbers, e.g., `[0, 1]` for the first two columns.
- **By Slice**: Use a range, e.g., `slice(0, 2)` for columns 0 and 1.
- **By Boolean Mask**: Use `True`/`False`, e.g., `[False, True, True]` for the second and third columns.
- **By Pattern**: Use a regex, e.g., columns starting with "A".
- **By Data Type (Include)**: Pick columns like numbers only.
- **By Data Type (Exclude)**: Skip columns like text.

### Example: Encoding Categories

One-hot encoding converts categorical columns into multiple binary columns (e.g., "male" = [1, 0], "female" = [0, 1]).

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

# One-hot encode 'Sex' and 'Embarked'
ct = make_column_transformer((OneHotEncoder(), ['Sex', 'Embarked']))
ct = make_column_transformer((OneHotEncoder(), make_column_selector(dtype_include=object)))
ct = make_column_transformer((OneHotEncoder(), make_column_selector(dtype_exclude='number')))
```

**Explanation**:
- Selects "Sex" and "Embarked" for encoding.
- Uses `make_column_selector` to automatically pick categorical (`object`) or exclude numerical (`number`) columns.

### Handling Columns: Passthrough and Drop

In `ColumnTransformer`, you can:
- **Passthrough**: Keep columns unchanged (e.g., if already preprocessed).
- **Drop**: Remove columns from the output (e.g., if irrelevant).

#### Example: Impute Missing Values, Passthrough, and Drop

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

# Sample dataset
X = pd.DataFrame({
    'A': [1, 2, np.nan],        # Numerical column with missing value
    'B': [10, 20, 30],          # Numerical column
    'C': [100, 200, 300],       # Numerical column
    'D': [1000, 2000, 3000],    # Numerical column (to drop)
    'E': [10000, 20000, 30000]  # Numerical column (to drop)
})

# Imputer for filling missing values
impute = SimpleImputer()

# Impute 'A', passthrough 'B' and 'C', drop 'D' and 'E'
ct = make_column_transformer(
    (impute, ['A']),               # Fill missing values in 'A'
    ('passthrough', ['B', 'C']),   # Keep 'B' and 'C' unchanged
    remainder='drop'               # Drop all other columns ('D' and 'E')
)

# Fit and transform the data
ct.fit_transform(X)
```

**Explanation**:
- **Impute**: Fills missing values in column "A" using the mean (default strategy for `SimpleImputer`).
- **Passthrough**: Keeps columns "B" and "C" unchanged.
- **Drop**: Removes columns "D" and "E" from the output, as they are not needed.

---

## 4. Fit vs. Transform Explained

Scikit-Learn uses two steps for preprocessing:
- **`fit`**: Learns from the data (e.g., finds the mean and standard deviation for scaling).
- **`transform`**: Applies what it learned (e.g., scales the data using the learned values).

### Why Split These?
- Prevents **data leakage**, where test data accidentally influences training.
- Ensures transformations are consistent across training and test sets.

#### Example: Scaling Numbers
- `fit`: Calculates mean and standard deviation from training data.
- `transform`: Uses those values to scale both training and test data.

Think of `fit` as learning the rules from training data, and `transform` as applying those rules to any data.

---

## 5. When to Use `fit_transform`

- Use **`fit_transform`** on **training data** to learn and apply transformations in one step.
- Use **`transform`** alone on **test data** to apply the same rules without relearning.

This avoids data leakage by ensuring the model does not see test data during training.

---

## 6. Why Choose Scikit-Learn for Preprocessing?

Scikit-Learn simplifies preprocessing because it:
- Supports **cross-validation** to fairly test the entire process.
- Allows **hyperparameter tuning** for preprocessing and models together.
- Protects data from accidental changes by separating `fit` and `transform`.
- Integrates well with pipelines for consistency and automation.

---

## 7. What is a Pipeline?

A **pipeline** combines preprocessing and modeling into one object. It ensures the same steps are applied to training and test data, preventing errors like data leakage. Pipelines simplify workflows, improve reproducibility, and make deployment easier.

### Example: Impute and Model

```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Sample training and test data
train = pd.DataFrame({'feat1': [10, 20, np.nan, 2], 'feat2': [25., 20, 5, 3], 'label': ['A', 'A', 'B', 'B']})
test = pd.DataFrame({'feat1': [30., 5, 15], 'feat2': [12, 10, np.nan]})

# Define features and target variable
features = ['feat1', 'feat2']
X, y = train[features], train['label']
X_new = test[features]

# Pipeline: Fill missing values, then predict
pipe = make_pipeline(SimpleImputer(), LogisticRegression())

# Train it
pipe.fit(X, y)

# Predict
pipe.predict(X_new)
```

**Explanation**:
- **Imputer**: Fills missing values in both training and test data.
- **Model**: Trains a logistic regression classifier on the preprocessed data.
- **Consistency**: Ensures the same preprocessing steps are applied to training and test sets.

---

## 8. Building and Managing Pipelines

Pipelines are powerful for combining preprocessing and modeling. Below are key techniques for managing pipelines.

### 8.1. Custom Features with `FunctionTransformer`

`FunctionTransformer` lets you add custom preprocessing rules, like clipping numbers or extracting parts of text.

#### Example: Clipping and Extracting

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Clip numbers between 100 and 600
clip_values = FunctionTransformer(np.clip, kw_args={'a_min': 100, 'a_max': 600})

# Get first letter of text
def first_letter(df):
    return df.apply(lambda x: x.str.slice(0, 1))

get_first_letter = FunctionTransformer(first_letter)
```

**Explanation**:
- **Clipping**: Limits numerical values to reduce outliers.
- **Text Extraction**: Creates new features from text data, useful for categorical modeling.

---

### 8.2. Adding Feature Selection to a Pipeline

Feature selection improves model performance by keeping only the most relevant features. Use `SelectPercentile` to retain top features based on statistical tests like chi-squared. Feature selection should occur after preprocessing but before model training.

#### Example: Pipeline with Feature Selection

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile, chi2

# Load Titanic dataset (predict survival based on passenger names)
df = pd.read_csv('http://bit.ly/kaggletrain')
X = df['Name']  # Text data (passenger names)
y = df['Survived']  # Target variable (0 = did not survive, 1 = survived)

# Vectorize text and apply logistic regression
vect = CountVectorizer()  # Convert text to numerical features
clf = LogisticRegression()  # Model for classification

# Pipeline without feature selection
pipe = make_pipeline(vect, clf)
print("Accuracy without feature selection:", cross_val_score(pipe, X, y, scoring='accuracy').mean())

# Add feature selection (keep top 50% of features)
selection = SelectPercentile(chi2, percentile=50)
pipe_with_selection = make_pipeline(vect, selection, clf)
print("Accuracy with feature selection:", cross_val_score(pipe_with_selection, X, y, scoring='accuracy').mean())
```

**Explanation**:
- **Without Feature Selection**: Uses all features from text vectorization, which may include irrelevant or noisy data.
- **With Feature Selection**: Keeps only the top 50% of features based on chi-squared scores, potentially improving efficiency and accuracy.
- **Cross-Validation**: Evaluates model performance using multiple data splits, ensuring reliable results.

Feature selection reduces model complexity and training time while maintaining or improving accuracy.

---

### 8.3. Visualizing Pipeline Diagrams (Scikit-Learn 0.23+)

You can create interactive diagrams of your pipeline in Jupyter Notebooks for better understanding. This is useful for debugging, sharing, or explaining workflows.

#### Example: Display Pipeline Diagram

```python
from sklearn import set_config
set_config(display='diagram')
pipe  # Displays an interactive diagram in Jupyter Notebooks

# Export diagram as HTML for sharing
from sklearn.utils import estimator_html_repr
with open('pipeline.html', 'w') as f:
    f.write(estimator_html_repr(pipe))
```

**Explanation**:
- **Interactive Diagram**: Click on pipeline steps to see details (e.g., parameters, inputs, outputs).
- **Export as HTML**: Saves the diagram for sharing with team members or documentation.

This feature enhances transparency and collaboration by visualizing the pipeline structure.

---

### 8.4. Accessing Feature Names from `ColumnTransformer` (Scikit-Learn 0.23+)

`get_feature_names()` now works with passthrough columns, making it easier to track feature names after transformation. This is helpful for understanding which features are used in the model and debugging.

#### Example: Get Feature Names

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd

# Load Titanic dataset (drop missing values for simplicity)
df = pd.read_csv('http://bit.ly/kaggletrain').dropna()
X = df[['Embarked', 'Sex', 'Parch', 'Fare']]  # Categorical and numerical columns

# Apply transformations
ct = make_column_transformer(
    (OneHotEncoder(), ['Embarked', 'Sex']),  # Encode categorical columns
    remainder='passthrough'                  # Keep numerical columns unchanged
)

# Fit and transform the data
ct.fit_transform(X)

# Get feature names
print(ct.get_feature_names_out())
```

**Explanation**:
- **One-Hot Encoding**: Converts "Embarked" and "Sex" into numerical columns (e.g., "Embarked_C", "Embarked_Q", "Sex_male").
- **Passthrough**: Keeps "Parch" and "Fare" unchanged.
- **Feature Names**: Lists all transformed feature names, making it easier to interpret model inputs.

This feature ensures you can trace features from raw data to model inputs, improving transparency.

---

### 8.5. Pipeline Slicing: Accessing Specific Steps

Use slicing to operate on parts of a pipeline, which is helpful for debugging or inspecting intermediate results. This allows you to check data after specific preprocessing steps.

#### Example: Access Pipeline Steps

```python
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Sample ColumnTransformer and pipeline
ct = ColumnTransformer([
    ('ohe', OneHotEncoder(), ['Sex']),             # Encode 'Sex'
    ('vectorizer', CountVectorizer(), 'Name'),     # Vectorize 'Name'
    ('imputer', SimpleImputer(), ['Age'])          # Impute 'Age'
])
fs = SelectPercentile(chi2, percentile=50)         # Feature selection
clf = LogisticRegression(solver='liblinear', random_state=1)  # Classification model

pipe = Pipeline([
    ('preprocessor', ct),          # Step 0: Preprocessing
    ('feature selector', fs),      # Step 1: Feature selection
    ('classifier', clf)            # Step 2: Classification
])

# Access specific steps
print(pipe[0].fit_transform(X))  # Step 0: preprocessor output
print(pipe[0:2].fit_transform(X, y))  # Steps 0 and 1: preprocessor and feature selector output
print(pipe[1].get_support())  # Step 1: feature selector (selected features)
```

**Explanation**:
- **Slicing**: `pipe[0]` accesses the first step (preprocessor), `pipe[0:2]` the first two steps (preprocessor and feature selector).
- **Intermediate Results**: Inspect transformed data after preprocessing or feature selection to ensure correctness.
- **Feature Selector Output**: `get_support()` shows which features were selected by `SelectPercentile`.

This technique helps debug pipelines and ensures each step works as expected.

---

## 9. Pipeline vs. `make_pipeline`

| Feature          | `Pipeline`                  | `make_pipeline`            |
|------------------|-----------------------------|----------------------------|
| **Naming Steps** | You name each step          | Names are automatic        |
| **Style**        | More detailed and clear     | Short and simple           |

### Example: Two Ways to Write It

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Creating a ColumnTransformer
ct = make_column_transformer((OneHotEncoder(), ['Embarked', 'Sex']), (SimpleImputer(), ['Age']), remainder='passthrough')

# Using make_pipeline (automatic naming)
pipe1 = make_pipeline(ct, LogisticRegression())

# Using Pipeline (explicit naming)
pipe2 = Pipeline([('preprocessor', ct), ('classifier', LogisticRegression())])
```

**Explanation**:
- **`Pipeline`**: Allows explicit step names, useful for debugging and clarity.
- **`make_pipeline`**: Automatically names steps, simpler for quick setups.

---

## 10. Checking Pipeline Steps

Use `named_steps` to inspect pipeline components and their results. This is helpful for verifying preprocessing or model behavior.

### Example: Inspecting Results

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

df = pd.read_csv('http://bit.ly/kaggletrain', nrows=6)
df = df[['Age', 'Pclass', 'Survived']]
X = df[['Age', 'Pclass']]
y = df['Survived']

pipe = make_pipeline(SimpleImputer(), LogisticRegression())
pipe.fit(X, y)

# Examine imputation statistics
pipe.named_steps['simpleimputer'].statistics_

# Examine model coefficients
pipe.named_steps['logisticregression'].coef_
```

**Explanation**:
- **Imputation Statistics**: Shows the values used to fill missing data (e.g., mean of "Age").
- **Model Coefficients**: Displays the weights assigned to features, useful for interpretation.

---

## 11. Controlling Columns in `ColumnTransformer`

- Use `'passthrough'` to keep columns unchanged.
- Use `'drop'` to remove columns from the output.

### Example: Mix and Match

```python
ct = make_column_transformer(
    (SimpleImputer(), ['Age']),         # Fill missing ages
    ('passthrough', ['Sex', 'Pclass']), # Keep these unchanged
    remainder='drop'                    # Drop everything else
)
```

**Explanation**:
- **Impute**: Fills missing values in "Age".
- **Passthrough**: Leaves "Sex" and "Pclass" unchanged.
- **Drop**: Removes all other columns not specified.

---

## 12. A Full Machine Learning Pipeline

Here is a reusable pattern for ML projects with structured data.

### Steps
1. **Load Data**: Split into features and labels.
2. **Preprocess**:
   - Numbers: Fill missing values and scale.
   - Categories: Fill missing values and encode.
3. **Select Columns**: Automate with `make_column_selector`.
4. **Build Pipeline**: Combine steps with a model.
5. **Evaluate**: Check performance with cross-validation.

### Example: Complete Workflow

```python
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Pick numerical and categorical columns
num_cols = make_column_selector(dtype_include='number')
cat_cols = make_column_selector(dtype_exclude='number')

# Preprocessing for numbers and categories
num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'), OneHotEncoder())

# Combine them
preprocessor = make_column_transformer(
    (num_pipeline, num_cols),
    (cat_pipeline, cat_cols)
)

# Add a model
full_pipeline = make_pipeline(preprocessor, LogisticRegression())

# Test it
cv_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring='accuracy')
print(f"Mean Accuracy: {cv_scores.mean():.2f}")
```

**Explanation**:
- **Column Selection**: Automatically picks numerical and categorical columns.
- **Preprocessing**: Handles missing values and scales numerical data, encodes categorical data.
- **Model**: Trains a logistic regression classifier.
- **Evaluation**: Uses cross-validation to assess model performance reliably.

---

## 13. Key Points to Remember

- **`ColumnTransformer`**: Handles different column types in one step.
- **Pipelines**: Make preprocessing and modeling repeatable and consistent.
- **Feature Selection**: Boosts performance by focusing on key features.
- **Data Leakage**: Avoid it by fitting only on training data.
- **Cross-Validation**: Ensures reliable evaluation of model performance.
- **Custom Transformations**: Use `FunctionTransformer` for tailored preprocessing.


