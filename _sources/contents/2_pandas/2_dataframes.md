# Pandas DataFrame

---

# 1. DataFrame Basics

## **What is a Pandas DataFrame?**
A **DataFrame** is a 2-dimensional (rows and columns) labeled data structure in the `pandas` library like a spreadsheet or SQL table. 

It consists of:
- **Rows** (observations): Indexed (default: 0-based integers, or custom labels).
- **Columns** (variables): Named (each column is a `Series` object).
- **Data**: Heterogeneous (can hold mixed types like integers, floats, strings).

It’s highly flexible, efficient, and equipped with powerful methods for data manipulation, cleaning, aggregation, and visualization.

---

# **2. Creating a DataFrame**
You can create a DataFrame from various sources: dictionaries, lists, NumPy arrays, CSV files, etc.

| Method               | Syntax                                    | Description                                |
|----------------------|------------------------------------------|--------------------------------------------|
| **Empty DF**        | `pd.DataFrame()`                        | Creates blank DataFrame                    |
| **From Single List** | `pd.DataFrame(list, columns=[], index=[])` | Creates single-column DF with custom labels |
| **From Nested List** | `pd.DataFrame([[1,2],[3,4]])`           | Creates DF from 2D list (rows → observations) |
| **From Dictionary**  | `pd.DataFrame({"A": [1,2], "B": [3,4]})` | Keys become column names                   |
| **From Existing DF** | `new_df = pd.DataFrame(original_df)`    | Creates independent copy                   |
| **From Files**       | `pd.read_csv("data.csv"`              | Real-world data loading                    |

Loading Data: 
`pd.read_.` 

❗ **File Handling Tip**: Always verify file paths to avoid `FileNotFoundError`

**From a NumPy Array**
```python
import numpy as np

array = np.array([[1, 2], [3, 4], [5, 6]])
df = pd.DataFrame(array, columns=["A", "B"])
print(df)
# Output:
#    A  B
# 0  1  2
# 1  3  4
# 2  5  6
```

--- 

# 3. DataFrame Attributes

A **DataFrame** is a predefined class in the pandas library that has several attributes providing useful metadata about the DataFrame object.

## Key Attributes
These tell you about a DataFrame’s structure:

| Attribute | Description | Example Output |
|-----------|-------------|----------------|
| `.columns` | Returns column names as an **Index object** | `Index(['Name', 'Age'], dtype='object')` |
| `.shape`   | Returns a tuple of **(rows, columns)** | `(1000, 5)` |
| `.size`    | Returns the total number of elements (rows × columns) | `5000` |
| `.dtypes`  | Displays data types of each column | `Name: object, Age: int64` |
| `.empty`   | Returns `True` if the DataFrame is empty, else `False` | `False` |
| `.index`   | Shows index details | `RangeIndex(start=0, stop=1000, step=1)` |
| `.values`  | Returns a NumPy array of all values | `[[1, 'Alice'], [2, 'Bob']]` |
| `.T`       | Transposes the DataFrame (swaps rows and columns) | Columns become row indices |

### **Example**
```python
print(df.shape)      # Output: (3, 2)
print(df.index)      # Output: RangeIndex(start=0, stop=3, step=1)
print(df.columns)    # Output: Index(['A', 'B'], dtype='object')
print(df.dtypes)     # Output: A    int64 \n B    int64
print(df.values)     # Output: [[1 2] [3 4] [5 6]]
```

❗ **Key Differences**:  
- `len(df)` ≠ `df.size`  
  - `len(df)` → Returns the number of rows (same as `shape[0]`)
  - `len(df.columns)` → Returns the number of columns (same as `shape[1]`)
  - `df.size` → Returns the total number of elements (rows × columns)

---

# 4. DataFrame Methods
A **DataFrame** is a predefined class in the pandas library, offering several methods that perform operations on the DataFrame and return results.

## Useful Methods
These help you explore and change DataFrames:

| Method       | Description | Key Parameters |
|-------------|--------------|-----------------|
| `.head()`    | Returns first **5 rows** | `n` - Custom row count (e.g., `head(3)`) |
| `.tail()`    | Returns last **5 rows** | `n` - Custom row count (e.g., `tail(3)`) |
| `.info()`    | Displays:<br>- Column count & names<br>- Non-null counts<br>- Data types<br>- Memory usage | `verbose` - Detailed output<br>`memory_usage` - Memory analysis |
| `.count()`   | Shows non-null values **per column** | `axis` - 0 for columns (default), 1 for rows |
| `.describe()` | Generates statistics:<br>- Count, mean, std<br>- Min/Max<br>- 25/50/75% quartiles | `include`/`exclude` - Control data types analyzed |
| `.nunique()` | Counts unique values **per column** | `axis` - 0 for columns (default), 1 for rows |

### Basic Data Exploration

```python
import pandas as pd

# Sample DataFrame
data = {'Temperature': [22.1, 23.5, None, 19.8],
        'City': ['Paris', 'London', 'Berlin', None]}
df = pd.DataFrame(data)

print("First 2 rows:")
print(df.head(2))

print("\nData Summary:")
print(df.describe(include='all'))

print("\nStructure Info:")
df.info(verbose=True)
```

### Smart `describe()` Filtering

```python
# Numerical columns only
df.describe(include=[np.number])

# Categorical columns only
df.describe(include=['object'])
```

## Changing Data Types

### Method 1: Change Data Type After Reading CSV
```python
drinks['beer_servings'] = drinks.beer_servings.astype(float)
```

### Method 2: Change Data Type While Reading CSV
```python
drinks = pd.read_csv('drinks.csv', dtype={'beer_servings': float})
```

### Bulk Type Conversion

```python
type_map = {'Temperature': 'float32',
            'City': 'category'}
df = df.astype(type_map)
```

---

📌 **Pro Tip**: Create method chains for efficient analysis  
`df.head(3).T.astype('str').to_dict()` → Quick JSON preview

---

# 5. Renaming Columns and Indexes
In pandas, we can rename or modify column names and indexes based on requirements. There are multiple ways to achieve this efficiently.

## Renaming Columns
- **Selective Rename**: for targeted changes.
    The `rename()` method allows you to rename specific columns using a dictionary:

    ```python
    df.rename(columns={"Name": "FullName"}, inplace=True)
    ```
    *❗ **Key Note:** Dictionary keys **must exactly match** existing column names. Non-matching keys are ignored silently.

- **Full Rename**: for complete renaming.
    You can replace all column names at once by assigning a list to the `columns` attribute:

    ```python
        # Assign new column names
        df.columns = ['new_col1', 'new_col2', 'new_col3']
    ```
    **Important:** The number of new names must match the number of existing columns; otherwise, a `ValueError` occurs.

    ```python
    # Incorrect: Mismatch in column count
    # df.columns = ['A', 'B']  # Raises ValueError
    ```
- **Case Conversion & Bulk Operations:**
  - `df.columns = df.columns.str.upper()` (Uppercase)
  - `df.columns = [col.upper() for col in df.columns]`
  - `df.columns = df.columns.str.replace(' ', '_')` (Underscore format)
  - `df.columns = [col + '_2023' for col in df.columns]`
  - ``

**Method Chaining:**
   ```python
   df = (pd.read_csv("data.csv")
           .rename(columns={"temp": "temperature"})
           .set_index("timestamp"))
   ```

**Validation Check Before Renaming:**
   ```python
   assert {'old_name', 'another_col'}.issubset(df.columns), "Missing columns!"
   ```

📌 **Pro Tip:** Use `df.filter()` to verify columns before renaming:
```python
valid_cols = df.filter(items=["temp", "hum"]).columns
df = df[valid_cols].rename(columns={"temp": "temperature"})
```

---

## Renaming Indexes
- **Selective**: for specific row labels.
  ```python
  # Rename specific index values
    df.rename(index={0: 'first', 1: 'second'}, inplace=True)

    # Function-based renaming
    df.rename(index=lambda x: f'row_{x+1}', inplace=True)
  ```
- **Full**: for full index replacement.
  ```python
  df.index = ["Person1", "Person2"]
  ```

---

## Understanding `axis`

- `axis=0`: Row-wise operations
- `axis=1`: Column-wise operations

- Dropping a Column
    `drinks.drop('continent', axis=1).head()`
- Calculating Mean for Each Column
    `drinks.mean(axis=0)  # Same as drinks.mean()`
- Calculating Mean for Each Row
    `drinks.mean(axis=1).head()`

---
## Adding and Dropping Columns

We can add new or drop columns to an existing DataFrame based on our requirements.

- Remove columns using the `drop()` method.

### 1. Inserting a Column at a Specific Position

You can insert a column at a specific index position using `insert()`:

```python
new_column = df['Product Cost'] * df['Quantity']
df.insert(5, "Total Cost", new_column)  # Position 5
```

🔧 **Pro Tip**: Use `assign()` for temporary column additions  
```python
df_temp = df.assign(Temp=lambda x: x.Price * 0.1)
```

### 2. Dropping a Column at a Specific Position
```python
# Single column
df.drop(columns='Customer Name', inplace=True)

# Multiple columns
df.drop(['Customer Name', 'Product Name'], axis=1, inplace=True)
```
## Dropping Rows

Use the `drop()` method to remove rows.
```python
# Single row by index
df.drop(3, axis=0, inplace=True)

# Multiple rows
df.drop([1, 2], axis=0, inplace=True)

# Conditional dropping
df = df[df['Sales'] > 1000]  # Keep rows where sales > 1000
```

---

# 6. `inplace` Parameter

Understand when and how to use `inplace` to modify DataFrames without creating new objects.

## Common Methods with `inplace`
```python
df.rename(columns=..., inplace=True)
df.drop(labels=..., inplace=True)
df.sort_values(by=..., inplace=True)
df.set_index(keys=..., inplace=True)
df.reset_index(inplace=True)
```

## Memory Management Tradeoffs
```python
# Good for large datasets (avoids duplication)
big_data.rename(columns={...}, inplace=True)  # Saves memory

# Bad for small data (unnecessary mutation)
small_data.dropna(inplace=True)  # Prefer: small_data = small_data.dropna()
```

⚠️ **Deprecation Warning**  
`inplace` parameter is being phased out in future pandas versions.  
**Recommended Alternative**:  
```python
# Instead of:
df.method(inplace=True)

# Use:
df = df.method()
```

---
# 7. Handling Missing Values

## 1. Understanding Missing Data

What is NaN (Not a Number)?
- **Purpose**: Represents missing or undefined numerical values.
- **Data Type**: Stored as `float64` in Pandas, even in non-float columns.
- **How NaNs are Created**:
  - Loading CSV/Excel files with empty cells.
  - Performing undefined mathematical operations.
  - Explicitly inserting `np.nan`.

## 2. Detecting Missing Values
- `.isna()`: Identifies `NaN` values
- `.isnull()`: Identifies `NaN` values
- `.isna().sum()`: Count missing values per column
- `.isna()..mean()*100`: Percentage of missing values

**Pro Tip**: Visualize missing values using the `missingno` library:
```python
import missingno as msno
msno.matrix(df)
```

## 3. Handling Missing Values (NaNs)
There are four common approaches to handle missing values:

1. **Drop rows** containing NaNs.
2. **Drop columns** containing NaNs.
3. **Fill NaNs** with imputed values.
4. **Use models** that natively handle NaNs.

## **Data Cleaning**
- `.dropna()`: Remove rows/columns with NaN.
- `.fillna(value)`: Replace NaN with a value.
- `.duplicated()`: Check for duplicate rows.
- `.drop_duplicates()`: Remove duplicates.

### 1. Removing Missing Values (`dropna()`)

```python
# Remove rows with any missing values
df_cleaned = df.dropna()

# Custom removal parameters
df.dropna(
    axis=0,         # 0=rows, 1=columns
    how='any',      # 'any' or 'all'
    thresh=2,       # Keep rows with ≥2 non-NA values
    subset=['Age']  # Only check specific columns
)
```

### 2. Imputing Missing Values (`fillna()`)
- **Numerical Data**: Mean/median for normally distributed data.
- **Categorical Data**: Replace with mode or "Unknown" category.
- **Time-Series Data**: Use forward/backward fill (`ffill`, `bfill`).

```python
# Basic imputation
df_filled = df.fillna({
    'Age': df['Age'].mean(),  # Replace NaN with column mean
    'Income': 0               # Replace NaN with 0
})

# Forward fill (propagate last valid value)
df.fillna(method='ffill', limit=1)

# Backward fill (propagate next valid value)
df.fillna(method='bfill')
```

### 3. Replacing Missing Values (`replace()`)
```python
# Replace NaN with a specific value
df_replaced = df.replace(np.nan, -999)

# Multi-value replacement
df.replace({
    np.nan: 'Missing',
    0: 'Zero'
})
```

### 4. Advanced Imputation Methods
If `SimpleImputer` is too basic, consider **KNNImputer** or **IterativeImputer**:
- `KNNImputer`: Uses K-nearest neighbors to estimate missing values.
- `IterativeImputer`: Uses a regression model to predict missing values based on other features.

```python
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

df = pd.read_csv('http://bit.ly/kaggletrain', nrows=6)
cols = ['SibSp', 'Fare', 'Age']
X = df[cols]

# Iterative Imputation
impute_it = IterativeImputer()
impute_it.fit_transform(X)

# KNN Imputation
impute_knn = KNNImputer(n_neighbors=2)
impute_knn.fit_transform(X)
```

---
## Imputing Missing Categorical Values

Two common methods:
1. Impute the most frequent value.
2. Impute the value `"missing"`, treating it as a separate category.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Sample data with missing categorical values
X = pd.DataFrame({'Shape': ['square', 'square', 'oval', 'circle', np.nan]})

# Impute with most frequent value
imputer = SimpleImputer(strategy='most_frequent')
print(imputer.fit_transform(X))

# Impute with a constant value "missing"
imputer = SimpleImputer(strategy='constant', fill_value='missing')
print(imputer.fit_transform(X))
```

---
## Adding a Missing Indicator
When imputing missing values, you can preserve information about which values were missing by adding a **missing indicator**.
Including a missing indicator can improve model performance when missing values are meaningful.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

X = pd.DataFrame({'Age': [20, 30, 10, np.nan, 10]})

# Impute missing values and add an indicator matrix
imputer = SimpleImputer(add_indicator=True)
imputer.fit_transform(X)
```

---

# **8. Accessing and Modifying Columns**

## Basic Column Selection

| Operation           | Syntax                    | Returns     |
|--------------------|-------------------------|------------|
| Single Column      | `df["col"]` or `df.col`  | `Series`   |
| Multiple Columns   | `df[["col1", "col2"]]`  | `DataFrame` |
| Conditional Rows   | `df[df["col"] > value]` | `DataFrame` |

---

## `iloc` Indexer (Position-Based Selection)

### Key Features
- Purely integer-based (row/column positions)
- Excludes the end index in ranges
- Cannot use column/row labels
- Optimized for positional access

### Syntax Patterns
- `df.iloc[row_index]`: return Single row
- `df.iloc[:, column_index]`: return Single column
- `df.iloc[row_range, column_range]`: Multiple rows and columns

```python
# Row and column selection
 df.iloc[0, 2]       # Row 0, Column 2
 df.iloc[:, 1:3]     # All rows, Columns 1-2
 df.iloc[2:5, :]     # Rows 2-4, All columns

# List of positions
 df.iloc[[0, 2, 4]]  # Rows 0, 2, 4
```

## `loc` Indexer (Label-Based Selection)

### Key Features
- Uses actual row/column labels
- Includes the end index in ranges
- Allows boolean indexing
- Requires a set index for meaningful row labels

### Syntax Patterns
- `df.loc["row_label"]`: return Specific row
- `df.loc[["row1", "row3"]]`: return Multiple rows
- `df.loc["row1":"row4"]`: return row1 to "row4" (Includes)

- `df.loc[:, "col_label"]`: return Specific column
- `df.loc[:, ["colA", "colC"]]`: return Multiple columns
- `df.loc[:, "colB":"colD"]`: return colB to "colD" (Includes)
- `df.loc[df["price"] > 100]`: return Boolean indexing

```python
# Set custom index
 df = df.set_index("product_id")

# Combined selection
 df.loc[["P100", "P102"], "price":"stock"]

# Conditional + column slice
 df.loc[df["category"] == "Electronics", "price":]
```

## Key Differences: `iloc` vs `loc`

| Feature        | `iloc`               | `loc`                 |
|---------------|---------------------|-----------------------|
| **Index Type** | Integer positions    | Labels/Booleans       |
| **Range End**  | Exclusive            | Inclusive             |
| **Column Ref** | Position only        | Name/Position         |
| **Performance**| Faster               | Slower (depends on index) |
| **Use Case**   | Positional access    | Label-based queries   |

### Boolean Indexing Best Practices
```python
# Complex conditions
 mask = (df["price"] > 100) & (df["stock"] < 50)
 df.loc[mask, ["product", "price"]]

# Lambda functions
 df.loc[lambda x: x["sales"] > x["target"]]
```

### Performance Optimization
```python
# Use iloc for large datasets
 large_df.iloc[10000:20000]  # Faster than loc

# Pre-calculate indexes
 fast_access = df.reset_index(drop=True)
```

## Common Pitfalls

### 1. Off-by-One Errors
```python
df.iloc[0:5]  # Rows 0-4 (5 rows)
df.loc["A":"E"]  # Includes E (5 rows)
```

### 2. Mixed Data Types
```python
# If index contains both strings and numbers
df.loc[1]  # Could return label 1 or position 1
```

### 3. Chaining Issues
```python
# Avoid:
df.iloc[5:10]["Price"]

# Use:
df.iloc[5:10, df.columns.get_loc("Price")]
```

---

# 9. Filtering

Filtering data is essential for extracting meaningful insights from large datasets.

Use boolean indexing or `.query()`.

## 1. Relational Operators
Filtering using relational operators such as `>`, `<`, `>=`, `<=`, `==`, and `!=`.
```python
# Single condition
high_cost = df['Product_Cost'] > 65000
filtered_df = df[high_cost]
```

## 2. Logical Operators
Filtering using logical operators such as AND (`&`), OR (`|`).

```python
movies[(movies.duration >= 200) & (movies.genre == 'Drama')]
```

```python
movies[(movies.genre == 'Crime') | (movies.genre == 'Drama') | (movies.genre == 'Action')]
```

## 3. Label-Based Filtering (`loc`)

```python
# Filtering specific products and customers
df_filtered = df.loc[(df.Product_Name == "iPhone 11") & (df.Customer_Name == "Shahid")]

# Selecting columns based on a condition
df_subset = df.loc[df['Sales'] > 500, ['Product', 'Region', 'Revenue']]
```

## 4. Position-Based Filtering (`iloc`)

```python
# Selecting first 5 rows
df_head = df.iloc[:5, :]

# Selecting specific rows and columns
critical_data = df.iloc[10:20, [0, 2, 4]]
```

## 5. Chained Filtering

```python
filtered = (
    df.query('Price > 1000')
      .loc[:, ['Product', 'Category']]
      .iloc[::2]  # Select every other row
)
```

## Advanced Filtering Methods

### 1. Multi-Value Selection with `isin()`

```python
# Single product filter
macbooks = df[df.Product_Name.isin(["Macbook Pro Laptop"])]

# Multiple products filter
df_premium = df[df.Product_Name.isin(["34in Ultrawide Monitor", "Macbook Pro Laptop"])]
```

### 2. Unique Value Analysis

```python
# Get unique products
unique_products = pd.unique(df.Product_Name)

# Identify rare customers
rare_customers = pd.unique(df.Customer_Name)[:10]
```

### 3. Handling Missing Data

```python
# Filtering non-null values
valid_emails = df[df.Email.notnull()]

# Combining null checks
complete_records = df[df['Address'].notnull() & df['Phone'].notnull()]
```

## Practical Example: E-Commerce Filtering

```python
import pandas as pd
import numpy as np

# Sample dataset
data = {
    'OrderID': range(1001, 1021),
    'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor'], 20),
    'Price': np.random.randint(200, 2000, 20),
    'Customer': np.random.choice(['Alice', 'Bob', 'Charlie', 'David'], 20),
    'Rating': np.round(np.random.uniform(3.5, 5, 20), 1)
}
df = pd.DataFrame(data)

# Complex filter
premium_tech = df[
    (df['Product'].isin(['Laptop', 'Monitor'])) &
    (df['Price'] > 1000) &
    (df['Rating'] >= 4.0) &
    (df['Customer'].notnull())
]

print("Premium Tech Products:")
print(premium_tech)
```

---

# 10. Sorting

Pandas provides powerful methods to sort DataFrames based on column values or index labels.

The `sort_values()` method sorts a DataFrame based on column values. By default, it sorts in ascending order for numerical values and alphabetically for strings.

The `sort_index()` method sorts a DataFrame based on index labels, often useful after reindexing or for time-series data.

## Syntax:
- `.sort_values(by="column_name", ascending=True)`: Sort by column(s).
- `.sort_index(ascending=True)`: Sort by index.

## Key Parameters
| Parameter  | Description                 | Default | Common Values  |
|------------|-----------------------------|---------|----------------|
| `by`       | Column(s) to sort by        | Required | String or list |
| `ascending`| Sort order                  | `True`  | `True`/`False` or list |
| `inplace`  | Modify original DataFrame   | `False` | `True`/`False` |
| `na_position` | NaN values placement   | 'last'  | 'first'/'last' |

## Usage Examples
```python
# Basic single-column sort
df_sorted = df.sort_values(by='Customer_Id')

# Descending order with NaN handling
df_sorted = df.sort_values(by='Revenue', ascending=False, na_position='first')

# Multi-column sorting
df_sorted = df.sort_values(by=['Department', 'Salary'], ascending=[True, False])
```

```python
# Basic index sort
df_sorted = df.sort_index()

# Descending index order
df_sorted = df.sort_index(ascending=False)

# Sort columns alphabetically
df_sorted = df.sort_index(axis=1)
```

## `sort_values()` vs `sort_index()`

| Feature         | `sort_values()`             | `sort_index()`           |
|----------------|-----------------------------|---------------------------|
| **Primary Use** | Sorts by data values       | Sorts by index labels    |
| **Multi-level** | Supports multi-column sorting | Supports multi-index sorting |
| **Axis Control** | Default: rows (`axis=0`)  | Can sort columns (`axis=1`) |
| **NaN Handling** | Placement control         | Follows label order       |

## Best Practices

1. **Reset Index After Sorting**
   ```python
   df = df.sort_values('date').reset_index(drop=True)
   ```
2. **Verify Sort Stability**
   ```python
   assert df.index.is_monotonic_increasing, "Data not sorted!"
   ```
3. **Optimize Memory Usage for Large Datasets**
   ```python
   df.sort_values(by='timestamp', inplace=True)
   ```
4. **Multi-Level Sorting**
   ```python
   df.sort_values(by=['department', 'salary'], ascending=[True, False], inplace=True)
   ```

## Common Pitfalls & Solutions

### 1. Mixed Data Types
```python
# Convert to consistent types before sorting
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
```

### 2. Case-Sensitive Sorting
```python
df.sort_values(by='Name', key=lambda x: x.str.lower())
```

---

# **11. Aggregation and Grouping**

GroupBy is a powerful operation in data analysis that allows you to split, apply, and combine data based on certain criteria. It is commonly used for summarizing data or performing aggregations.

## Steps in GroupBy Operation

1. **Splitting**: The data is split into groups based on a specified criterion.
2. **Applying**: A function is applied to each group independently.
3. **Combining**: The results from all groups are combined to form a new DataFrame.

## **GroupBy**
- The `groupby()` method is a predefined method in the DataFrame class. 
- We should access this method by using DataFrame object.
- It returns a `GroupBy` object, which can be used to perform various operations on the grouped data.

- `.groupby(column)`: Group data for aggregation.

### Key Parameters
| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `by` | Grouping criteria | Required | `by='Department'` |
| `axis` | Group rows(0) or columns(1) | 0 | `axis=1` |
| `as_index` | Use group labels as index | True | `as_index=False` |
| `sort` | Sort group keys | True | `sort=False` |

### 1. Grouping by a Single Column
Group the DataFrame by the `Product` column and calculate the sum for each group:

```python
# Group by product and sum sales
product_group = df.groupby('Product')
total_sales = product_group['Sales'].sum()
```

### 2. Multi-Column Grouping
```python
# Group by product and region
regional_sales = df.groupby(['Product', 'Region'])['Sales'].mean()
```

## **Aggregation**
- `.sum()`, `.mean()`, `.median()`, `count()`,  `.min()`, `.max()`, etc.

### 1. Custom Aggregations
```python
# Multiple statistics at once
product_stats = df.groupby('Product')['Sales'].agg(['sum', 'mean', 'count'])
```

### 2. Named Aggregations
```python
df.groupby('Product').agg(
    total_sales=('Sales', 'sum'),
    avg_sales=('Sales', 'mean'),
    orders=('Sales', 'count')
)
```

🔗 **Pro Tip**: Combine with `sort_values` for ordered analysis  
`df.groupby('Category')['Sales'].sum().sort_values(ascending=False).head(5)`

---

# **12. Merging and Joining**

**Methods**
- `.merge()`: SQL-like joins.
- `.concat()`: Stack DataFrames.
- `.join()`: Join on index.

## Merges

Merging or joining is the process of combining two DataFrames based on common attributes in columns. This operation is similar to the `JOIN` operation in databases.

The `merge()` function in pandas enables various types of join operations between DataFrames.

### Syntax:
- `pd.merge(df1, df2, on="column", how="join_type")`: 
- **Parameters**:
  - `df1`: The first DataFrame.
  - `df2`: The second DataFrame.
  - `on`: The column(s) to join on.
  - `how`: The type of join (default is 'inner').

Practical Examples
```python
import pandas as pd

customers = pd.DataFrame({
    'cust_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'cust_id': [1, 2, 4],
    'total': [150.0, 99.99, 200.0]
})
```

### **Types of Joins**
#### 1. Inner Join (Default)
- Common data only.
```python
pd.merge(customers, orders, on='cust_id')
```
#### 2. Left Join
- A **left join** keeps all rows from the left DataFrame and fills missing data from the right DataFrame with `NaN`.

```python
pd.merge(customers, orders, on='cust_id', how='left')
```
#### 3. Right Join
- A **right join** keeps all rows from the right DataFrame and fills missing data from the left DataFrame with `NaN`.

```python
pd.merge(customers, orders, on='cust_id', how='right')
```
#### 3. Outer Join
- All rows from both DataFrames, missing values filled with `NaN`.
```python
pd.merge(customers, orders, on='cust_id', how='outer')
```

**Join Types**:
- `pd.merge(customers, orders, on='cust_id', how='outer', validate='one_to_one')`
  - **One-to-One**: One row in the left and right DataFrames match.
  - **Many-to-One**: Duplicate values in the left DataFrame.
  - **Many-to-Many**: Duplicate values in both DataFrames.

---
### Advanced Techniques

1. Multi-Key Merges
    ```python
    pd.merge(df1, df2, on=['country', 'city'])
    ```
2. Indicator Flag
    ```python
    pd.merge(df1, df2, how='outer', indicator=True)
    ```
3. Merging on Indexes
    ```python
    pd.merge(df1, df2, left_index=True, right_index=True)
    ```

---

## Concatenation with `pd.concat()`

Concatenation in pandas allows us to combine or stack multiple DataFrames either vertically or horizontally based on specific requirements.

## The `concat()` Function

The `concat()` function in pandas is used to concatenate DataFrames along a particular axis (either rows or columns).

- `.concat()`: Stack DataFrames.

### Core Syntax
```python
pd.concat(
    objs,             # List/sequence of DataFrames
    axis=0,           # 0=vertical (default), 1=horizontal
    join='outer',     # 'outer' (default) or 'inner'
    ignore_index=False # Reset index after concatenation
)
```

### Concatenation Modes

#### 1. Vertical Concatenation (Axis=0)
- Stacks DataFrames one below the other.

```python
import pandas as pd

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

result = pd.concat([df1, df2])
```
#### 2. Horizontal Concatenation (Axis=1)
- Aligns DataFrames side by side.

**Example:**
```python
df3 = pd.DataFrame({'C': [9, 10], 'D': [11, 12]})
result = pd.concat([df1, df3], axis=1)
```

### 3. Mixed Axis Concatenation
```python
# Combine vertical and horizontal
combined = pd.concat([
    pd.concat([df1, df2], axis=0),
    pd.concat([df3], axis=1)
], axis=1)
```

🔗 **Pro Tip**: Combine with `groupby` for complex data assembly  
```python
pd.concat([group for _, group in df.groupby('category')])
```
---

# 13. Concatenating Multiple CSV Files

In real-world scenarios, data is often spread across multiple CSV files. To analyze or process the data efficiently, we need to **concatenate these files** into a single dataset.

## ✅ Steps to Concatenate CSV Files

### 1. Using the `glob` Module
We can use the `glob` module to find CSV files in a more structured way:

```python
import glob
import pandas as pd

path = "./data"
csv_files = glob.glob(os.path.join(path, "*.csv"))
```
This fetches **all CSV files** from the directory in a list format.

### 2. **Concatenating All CSV Files**

Once we have the list of CSV files, we can use the `pandas` library to read and merge them into a single DataFrame.

```python
# Read and concatenate all CSV files
result = (pd.read_csv(file) for file in csv_files)
df = pd.concat(result, ignore_index=True)

print(df.head())
```
- `pd.read_csv(file)`: Reads each file into a DataFrame.
- `pd.concat()`: Combines all DataFrames into one.
- `ignore_index=True`: Resets the index after concatenation.

### Method 2: `os` module

1️. **Accessing All Files in a Directory**

The `os` module is a built-in Python library that allows interaction with the operating system, such as listing files in a directory.

```python
import os

path = "./data"
all_files = os.listdir(path)  # Retrieves all file names in the folder
print(all_files)
```
This will return a list of all files inside the `./data` directory.

2️. **Filter Only CSV Files**
Since the directory may contain various file types, we must filter only CSV files.

```python
csv_files = [file for file in all_files if file.endswith(".csv")]
print(csv_files)
```
This ensures that only `.csv` files are selected for concatenation.

---

# **14. Applying Functions**
- `.apply()`: Apply a function along an axis.
- `.map()`: Apply to a Series (column).
- `.applymap()`: Element-wise for entire DataFrame.

## `.apply()`
### 1. `apply()` for Series-Wide Operations
```python
# Calculating string length for each row in 'Name' column
train['Name_length'] = train.Name.apply(len)
```

### 2. Adding a Column Using `apply()`

The `apply()` method allows row-wise operations to calculate a new column:

```python
def calculate_total(row):
    return row['Product Cost'] * row['Quantity']

df['Total Cost'] = df.apply(calculate_total, axis=1)
```

### 3. `apply()` for DataFrames
```python
# Finding max value per column
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=0)
```

```python
# Finding max value per row
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=1)
```

---
## `.map()`: Apply to a Series (column).

### 1. `map()` for Simple Mappings
```python
# Mapping categorical values to numerical
train['Sex_num'] = train.Sex.map({'female': 0, 'male': 1})
```

---

## `.applymap()`: Element-wise for entire DataFrame.

### 4. `applymap()` for Element-Wise Operations
```python
# Applying function to every element
drinks.loc[:, 'beer_servings': 'wine_servings'] = drinks.loc[:, 'beer_servings': 'wine_servings'].applymap(float)
```

---

# 15. 📅 Working with Date and Time

Date and time manipulations are crucial when working with datasets that contain time-based information.

## Converting Data Type
### 1. Loading a CSV with Date Parsing
When loading a CSV file, if a column contains date values, Pandas treats it as an **object** by default. To explicitly parse date columns, use `parse_dates` while reading the CSV file.

```python
df = pd.read_csv('sales7_dates.csv', parse_dates=['Pur_Date'])
```

### 2. Converting Object Data Type to Date
Sometimes, date values are stored as objects (strings). We can convert them explicitly using:

- `pd.to_datetime(df['date_col'], format='%d%m%Y')`
- `.astype('datetime64[ns]')`

```python
df['Pur_Date'] = pd.to_datetime(df['Pur_Date'])

df['Pur_Date'] = df['Pur_Date'].astype('datetime64[ns]')
```

**Formatting Date Strings**
Date formats can vary, such as **"03-23-15"** or **"3|23|2015"**. We can use the `format` parameter to specify the exact format.
```python
df['PurDate'] = pd.to_datetime(df['PurDate'], format='%d%m%Y')
df['PurDate'] = pd.to_datetime(df['PurDate'], format='%d%b%Y')  # Example: 23Mar2015
```

**Handling Missing Dates (NaT Values)**
If your date column contains missing values (`NaN`), converting them to datetime will raise an error. Use `errors="coerce"` to convert invalid values to **NaT (Not a Time).**
```python
df['PurDate'] = pd.to_datetime(df['PurDate'], errors="coerce")
```

## Selecting Date Ranges
We can filter data based on a **start and end date.**

### Selecting Data Between Two Dates
```python
start = df['Pur_Date'] > '2019-1-1 01:00:00'
end = df['Pur_Date'] < '2019-1-1 05:00:00'
result = df[start & end]
```

---

## Accessing Last N Days, Months, or Years
Pandas allows retrieving records for the last **N days, months, or years** by setting the date column as the index and using the `.last()` method.

### Selecting Recent Date Ranges
```python
df = df.set_index("Pur_Date")

# Last 10 days
days_10 = df.last("10D")

# Last 40 days
days_40 = df.last("40D")

# Last 1 month
month_1 = df.last("1M")

# Last 1 year
year_1 = df.last("1Y")
```

---

## Extracting Date Components
We can extract individual components of the date (year, month, day, hour, minute) using `.dt`.

### Extracting Year, Month, Day, Hour, and Minute
```python
df['year'] = df['Pur_Date'].dt.year
df['month'] = df['Pur_Date'].dt.month
df['day'] = df['Pur_Date'].dt.day
df['hour'] = df['Pur_Date'].dt.hour
df['minute'] = df['Pur_Date'].dt.minute
```

---

## Encoding Days of the Week
Extracting the **day of the week** helps in analyzing trends, such as comparing sales on different days.
Use `.dt.day_name()` or `.dt.weekday` to get the day of the week.

### Getting the Day of the Week
```python
df['PurDate'] = pd.to_datetime(df['PurDate'])
print(df["PurDate"].dt.day_name())  # Monday, Tuesday, etc.
```

### Getting the Day as a Number (Monday = 0, Sunday = 6)
```python
print(df['PurDate'].dt.weekday)
```

---

# **16. Tools and Utilities**
- **`pd.read_*`**: Read from CSV, Excel, JSON, SQL, etc.
  ```python
  # df = pd.read_excel("file.xlsx")
  ```
- **`.to_*`**: Export to formats.
  ```python
  df.to_csv("output.csv", index=False)
  ```
- **`.value_counts()`**: Count unique values in a column.
  ```python
  print(df["Department"].value_counts())  # Output: HR    2 \n IT    2
  ```
- **`.crosstab()`**: Cross-tabulation.
  ```python
  print(pd.crosstab(df["Department"], df["Salary"]))
  ```

