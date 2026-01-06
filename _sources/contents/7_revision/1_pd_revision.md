
# Series Attributes and Methods

## Series Attributes
A **Series** is a predefined class in Python, and it has several important attributes that provide useful information about the object.

### Key Attributes:
- **values**: Returns the values of the series as a NumPy array.
- **index**: Returns the index range of the series, like `RangeIndex(start=0, stop=6, step=1)`.
- **dtypes**: Returns the data type of the series' values.
- **size**: Returns the number of elements in the series.

---

## Series Methods
The **Series** class also has various predefined methods to perform operations on the series of values.

### Common Methods:
- **head()**: Returns the first five values of the series.
- **tail()**: Returns the last five values of the series.
- **sum()**: Returns the sum of all the values in the series.
- **count()**: Returns the number of non-NaN/null values in the series.
- **mean()**: Returns the mean (average) of the series' values.
- **describe()**: Returns summary statistics, such as count, mean, std, min, 25%, 50%, 75%, and max.
- **unique()**: Returns the unique values from the series.
- **nunique()**: Returns the number of unique values in the series.

---


# DataFrame in Pandas

## What is a DataFrame?
A **DataFrame** is a two-dimensional, size-mutable, and potentially heterogeneous tabular data structure. It can be thought of as a table with rows and columns, similar to a spreadsheet or SQL table.

### Key Points:
- **Two-dimensional structure**: A DataFrame has rows and columns.
- **Table-like structure**: It is similar to a table used in databases or spreadsheets.

---

## Creating a DataFrame
A DataFrame is a predefined class in the pandas library, and there are several ways to create one:

### Common Ways to Create a DataFrame:
1. **Empty DataFrame**:
   ```python
   pd.DataFrame()
   ```
2. **Using a single list**:
   ```python
   pd.DataFrame(l, columns=[cols], index=i)
   ```
3. **Using a nested list**:
   ```python
   pd.DataFrame(l)
   ```
4. **Using a dictionary**:
   ```python
   pd.DataFrame(d)
   ```
5. **Using another DataFrame**:
   ```python
   # Example of creating a new DataFrame from an existing one
   new_df = pd.DataFrame(existing_df)
   ```
6. **Loading from Files**:
   - **CSV file**: 
     ```python
     pd.read_csv("file.csv")
     ```
   - **JSON file**:
     ```python
     pd.read_json("file.json")
     ```
   - **Excel file**:
     ```python
     pd.read_excel("file.xlsx")
     ```
   - **TSV file**:
     ```python
     pd.read_table("file.tsv")
     ```
   - **Table from a webpage**:
     ```python
     pd.read_html("url")
     ```

---

## Accessing Data from a DataFrame

### Accessing a Single Column:
To access a single column, use either:
- Using dot notation:
  ```python
  df.column
  ```
- Using bracket notation:
  ```python
  df["column"]
  ```
Both will return the column as a **Series**.

### Accessing Multiple Columns:
To access multiple columns, pass a list of column names:
```python
df[["col1", "col2"]]
```
This will return a **DataFrame** with the selected columns.

---


# DataFrame Attributes

A **DataFrame** is a predefined class in the pandas library that has several attributes that provide information about the DataFrame object.

### Key DataFrame Attributes:

- **columns**: 
  - Returns all column names from the DataFrame.
  
- **shape**: 
  - Returns the total number of rows and columns as a tuple `(rows, columns)`.

  - **shape[0]**: 
    - Returns the total number of rows in the DataFrame.
  
  - **shape[1]**: 
    - Returns the total number of columns in the DataFrame.

- **size**: 
  - Returns the total number of elements (values) in the DataFrame.

- **dtypes**: 
  - Returns the data type of each column.

- **empty**: 
  - Checks if the DataFrame is empty. 
  - Returns `True` if the DataFrame is empty, otherwise `False`.

- **index**: 
  - Returns the start and end values of the DataFrame’s index.

- **values**: 
  - Returns the values of the DataFrame as a 2D array, with each row’s values in one array from the first to the last row.

- **T (Transpose)**: 
  - Returns the transposed DataFrame, where rows become columns and columns become rows.

---

### Length of DataFrame:
To get the total number of rows in the DataFrame, you can use the `len()` function:

```python
len(df)
```

This will return the total number of rows in the DataFrame.

---

### Summary of Attributes:
- **columns**: Column names of the DataFrame.
- **shape**: Dimensions of the DataFrame, `(rows, columns)`.
- **size**: Total number of elements in the DataFrame.
- **dtypes**: Data types of each column.
- **empty**: Checks if the DataFrame is empty.
- **index**: Start and end values of the index.
- **values**: DataFrame values as a 2D array.
- **T**: Transposes the DataFrame.

---


# DataFrame Methods

A **DataFrame** is a predefined class in the pandas library, offering several methods that perform operations on the DataFrame and return results.

### Key DataFrame Methods:

- **head()**:
  - Returns the first five rows of the DataFrame.

- **tail()**:
  - Returns the last five rows of the DataFrame.

- **info()**:
  - Provides summary information about the DataFrame:
    - Type of object
    - Range of object
    - Number of columns
    - Number of rows
    - Data type of each column
    - Number of data types
    - Total memory usage

- **count()**:
  - Returns the number of non-null values in each column.

- **describe()**:
  - Returns descriptive statistics of the DataFrame, including:
    - count, mean, std, min, 25%, 50%, 75%, max

- **nunique()**:
  - Returns the number of unique values for each column in the DataFrame.

- **astype(p)**:
  - Returns the data type values of the DataFrame, where `p` is the desired data type.

### Rearranging Columns in DataFrame:
You can rearrange the columns in a DataFrame to customize their order. For example:

```python
df[["col5", "col2"]]
```

This will rearrange the columns in the specified order.

---

### Summary of Methods:
- **head()**: First five rows.
- **tail()**: Last five rows.
- **info()**: Summary of the DataFrame.
- **count()**: Number of non-null values.
- **describe()**: Descriptive statistics.
- **nunique()**: Number of unique values per column.
- **astype(p)**: Change or return data types of columns.
- **Rearranging columns**: Custom order of columns using `df[["col1", "col2"]]`.

---


# Renaming Columns & Index in DataFrame

In pandas, we can rename or modify the column names and index based on the requirement. There are different ways to accomplish this.

## Renaming Columns

### 1. Using `rename()` Method

You can change one or multiple column names using the `rename()` method. This method accepts a Python dictionary, where:

- The **key** is the old column name.
- The **value** is the new column name.

```python
df.rename(columns={'old_col': 'new_col'}, inplace=True)
```

**Note**:
- The key (old column name) must match the column name in the DataFrame, otherwise, the change will not be applied.

### 2. Using `columns` Attribute

Another way to change column names is by directly modifying the `columns` attribute. You can assign a list of new column names to the `columns` attribute.

```python
df.columns = ['new_col1', 'new_col2', 'new_col3']
```

**Note**:
- When using the `columns` attribute to change column names, the number of new names must match the number of existing columns. Otherwise, a `ValueError: Length mismatch` will occur.

## Renaming Index

You can also change the index of a DataFrame:

### 1. Using `rename()` Method for Index

Just like with columns, you can rename the index using the `rename()` method by passing the new index values:

```python
df.rename(index={0: 'row1', 1: 'row2'}, inplace=True)
```

### 2. Using `index` Attribute

You can directly modify the `index` attribute to change the index:

```python
df.index = ['row1', 'row2', 'row3']
```

## Converting Column Names to Upper Case

To quickly convert all column names to uppercase, you can use the `str.upper()` method on the `columns` attribute:

```python
df.columns = df.columns.str.upper()
```

---

### Summary:
- **Renaming Columns**: 
  - Use `rename(columns={'old': 'new'})` for individual columns.
  - Use `df.columns = ['new1', 'new2']` for multiple columns.
- **Renaming Index**: 
  - Use `rename(index={'old': 'new'})` for individual index labels.
  - Use `df.index = ['new1', 'new2']` for multiple index labels.
- **Convert Columns to Uppercase**: `df.columns = df.columns.str.upper()`.

---


# The `inplace` Parameter in pandas

The `inplace` parameter is used in various DataFrame methods to specify whether changes should be made to the original DataFrame or if a new DataFrame should be returned.

## What is `inplace`?

- **In-place operation** means modifying the original object directly rather than creating a new one.
  
## How Does `inplace` Work?

The `inplace` parameter is commonly used in methods such as:
- `rename(p, inplace)`
- `drop(p, inplace)`
- `sort_values(p, inplace)`
- `set_index(p, inplace)`

### Values of `inplace`

- `inplace=False`: By default, pandas will **not** modify the original DataFrame. It will return a new object with the changes.
- `inplace=True`: The original DataFrame will be modified, and no new DataFrame is returned.

### Example:

```python
df.rename(columns={'old_col': 'new_col'}, inplace=True)
```

In this example, the `rename()` method updates the original `df` DataFrame, and no new DataFrame is returned.

---

## Summary:
- **`inplace=False`**: Returns a new DataFrame with changes.
- **`inplace=True`**: Modifies the original DataFrame.

---


# Handling Missing or NaN Values

## NaN Value
- **NaN** stands for **Not a Number**.
- NaN represents missing values in data.
- The data type of NaN is `float`.
- When loading a CSV file, missing values are automatically considered as NaN.
- For example, if a user doesn't share their income or address in a survey, those fields might be marked as missing (NaN).

## None vs. NaN
- **None**: A Python object that represents "nothing" or a null value.
- **NaN**: A Pandas-related object specifically used to represent missing or undefined data in a dataset.

## Checking for NaN Values

### `isna()` and `isnull()` Methods
- These methods check if there are missing (NaN) values in the DataFrame.
- If missing values exist, they return `True`; otherwise, they return `False`.
- Both methods work the same way, so you can use either one.

### `notnull()` Method
- This method checks for missing values in a DataFrame.
- If there are missing values, it returns `False`; otherwise, it returns `True`.

## Counting NaN Values by Column
- You can count the number of missing values in each column of the DataFrame.
- Use `isna()` combined with `sum()` to count missing values for each column.

## Handling Missing Values

### Dropping Rows with Missing Values
- `dropna()` drops rows where at least one value is missing.
- `dropna(inplace=True)` modifies the existing DataFrame and drops the rows with missing values.

### Filling Missing Values
- `fillna(p)` allows you to fill NaN values with a specific value (e.g., 0, mean, or median).
- `replace(np.nan, num)` can be used to replace NaN values with a specific number.

---


# Selecting Data with `iloc[]` and `loc[]`

## Selecting Columns
- **Single Column**:  
  Use `df.column` or `df["column"]` to select a single column. This returns a Series.
  
- **Multiple Columns**:  
  Use `df[["col1", "col2"]]` to select multiple columns. This returns a DataFrame.
  
- **Specific Column Values**:  
  Use `df[df["column"] == "value"]` to filter rows based on a specific condition in the column.

---

## `iloc[]` and `loc[]` Indexers
- **`iloc[]`** and **`loc[]`** are indexers used to select rows and columns from a DataFrame.
  - `iloc[]`: Index-based selection (integer position).
  - `loc[]`: Label-based selection (name-based index).

---

## `iloc[]` Indexer
- `iloc[]` is used for integer-location-based indexing, where we pass integer positions to select rows and columns.
  
### Key Points:
- **Selection Method**: We can select rows and columns using integer positions.
- **Exclusion**: `iloc[]` excludes the last element in the selection (exclusive end).
  
### Syntax:
- **Single Row or Column**:  
  `df.iloc[<row_selection>]` or `df.iloc[:, <col_selection>]`
  
  Examples:  
  `df.iloc[0]`, `df.iloc[-1]`, `df.iloc[:, 0]`, `df.iloc[:, -1]`
  
- **Multiple Rows and Columns**:  
  `df.iloc[<row_range>, <col_range>]`
  
  Examples:  
  `df.iloc[0:5]`, `df.iloc[:, 0:2]`, `df.iloc[0:5, 0:3]`

---

## `loc[]` Indexer
- `loc[]` is used for label-based indexing, where we specify the labels (names) of the rows and columns.
- It also supports **boolean indexing**.

### Key Points:
- **Selection Method**: We pass the labels of rows and columns.
- **Inclusion**: `loc[]` includes the last element in the selection (inclusive end).
  
### Syntax:
- **Selecting Rows and Columns by Label**:  
  `df.loc[<row_selection>, <col_selection>]`

---

### Using `loc[]` for Row Selection by Label
- You can set a custom index using `set_index()`. After that, you can directly access rows based on the labels.

Example:  
```python
df.set_index("Product name", inplace=True)
df.loc["ThinkPad Laptop"]  # Access row by label
```

- **Multiple Rows**:  
  `df.loc[['iPhone 9', 'iPhone 11']]` selects multiple rows.

- **Selecting Specific Columns**:  
  `df.loc[['iPhone 9', 'iPhone 11'], ['Product cost', 'Customer id']]`  
  `df.loc[['iPhone 9', 'iPhone 11'], 'Order id' : 'Product cost']`

---

## Boolean / Logical Indexing
- Boolean indexing allows you to select rows based on specific conditions.
- It’s one of the most commonly used techniques in data analysis.

### Example:
- **Single Condition**:
  ```python
  a = df1['Product name'] == 'LG Washing Machine'
  df2 = df1.loc[a]
  ```
- **Condition with Specific Columns**:
  ```python
  df2 = df1.loc[a, 'Order id' : 'Product cost']
  ```

---


# DataFrame Filtering

Filtering data in a DataFrame is a common task in data analysis. It is essential for narrowing down datasets based on specific conditions, which helps in extracting meaningful insights.

## Filtering Examples
- **Banking**: Select all the active customers whose accounts were opened after 1st January 2020.
- **Organization**: Fetch information of employees who have spent more than 3 years in the organization and received the highest rating in the past 2 years.
- **Telecom**: Analyze complaints data and identify customers who filed more than 5 complaints in the last year.

---

## Methods of Filtering Data
You can filter data in a DataFrame using different approaches:

### 1. Relational Operators
You can filter data by applying relational operators such as `>`, `<`, `>=`, `<=`, `==`, and `!=`. You can use these operators for single or multiple conditions.

#### Example:
```python
con1 = df1['Product_Cost'] > 65000
df2 = df1[con1]
```

- **Multiple Conditions**:
```python
con1 = df1['Product_Cost'] > 50000
con2 = df1['Product_Cost'] < 60000
df2 = df1[con1 & con2]
```

### 2. Using `loc[]` and `iloc[]` Indexers
You can also filter data using `loc[]` and `iloc[]` indexers.

#### Example using `loc[]`:
```python
con1 = df1.Product_Name == "iPhone 11"
con2 = df1.Customer_Name == "Shahid"
df2 = df1.loc[con1 & con2]
```

#### Example using `iloc[]`:
```python
df2 = df1.iloc[:5, ]
```

### 3. Filtering by Row Position and Column Name
You can select data by specifying row positions and column names.

#### Example:
```python
rows = df1.index[0:]
cols = ["Product_Name", "Customer_Id"]
df2 = df1.loc[rows, cols]
```

- **Selecting a Subset of Rows and Columns**:
```python
rows = df1.index[0:4]
cols = ["Product_Name", "Customer_Id"]
df2 = df1.loc[rows, cols]
```

### 4. Selecting Multiple Values in a Column
You can filter the DataFrame by providing multiple values for a column.

#### Example:
```python
a = df1.Product_Name == "LG Washing Machine"
b = df1.Customer_Id == 1
c = a | b
df2 = df1.loc[c]
```

### 5. `isin()` Method
The `isin()` method is used to filter data based on a list of values. This method is available in the Series class.

#### Example:
```python
a = ["Macbook Pro Laptop"]
b = df1.Product_Name.isin(a)
df2 = df1[b]
```

- **Multiple Values**:
```python
a = ["34in Ultrawide Monitor", "Macbook Pro Laptop"]
b = df1.Product_Name.isin(a)
df2 = df1[b]
```

---

## Selecting Non-Missing Data

### 1. `notnull()` Method
The `notnull()` method helps you select rows that do not have missing (NaN) values. This method is available in the Series class.

#### Example:
```python
d = df1.column.notnull()
df2 = df1[d]
```

---

## Summary
- Filtering data is essential for extracting insights based on specific conditions.
- You can use relational operators, `loc[]`, `iloc[]`, `isin()`, and `notnull()` methods to filter data in various ways.
- `isin()` is useful for filtering based on multiple values in a column.
- `notnull()` is used to filter non-missing data from a DataFrame.

---


# Sorting in DataFrame

Sorting is a common operation in data analysis, where you need to sort data by columns, indexes, or values to organize and better understand the dataset.

## Methods for Sorting

### 1. `sort_values()` Method

The `sort_values()` method is used to sort the values in a DataFrame based on a specific column. By default, the sorting is in ascending order for numerical values and alphabetical order for strings.

#### Syntax:
```python
df.sort_values(by="column_name", ascending=True)
```

- **Parameters**:
  - `by`: Specifies the column by which to sort the DataFrame.
  - `ascending`: Defines the sorting order. Default is `True` (ascending). Set to `False` for descending order.

#### Example 1: Sort by a single column
```python
df2 = df1.sort_values(by="Customer_Id")
```

#### Example 2: Sort in descending order
```python
df2 = df1.sort_values(by="Customer_Id", ascending=False)
```

#### Example 3: Alternate way to specify descending order
```python
df2 = df1.sort_values(by="Customer_Id", ascending=0)
```

### 2. `sort_index()` Method

The `sort_index()` method is used to sort the index of the DataFrame. It helps in organizing the DataFrame based on the index values.

#### Syntax:
```python
df.sort_index(ascending=True)
```

- **Parameters**:
  - `ascending`: Defines whether to sort in ascending (`True`) or descending (`False`) order. Default is `True`.

#### Example: Sort by index
```python
df2 = df1.sort_index()
```

---

## Summary
- **`sort_values()`**: Sorts the DataFrame based on column values, either in ascending or descending order.
- **`sort_index()`**: Sorts the DataFrame based on its index values.
- Both methods help in organizing and structuring data for analysis.

---


# GroupBy in DataFrame

GroupBy is a powerful operation in data analysis that allows you to split, apply, and combine data based on certain criteria. It is commonly used for summarizing data or performing aggregations.

## What is GroupBy?

GroupBy is used to split the data into different groups based on some criteria, apply a function to each group, and combine the results into a DataFrame. It is one of the most commonly used operations in data analysis.

### Steps in GroupBy Operation

1. **Splitting**: The data is split into groups based on a specified criterion.
2. **Applying**: A function is applied to each group independently.
3. **Combining**: The results from all groups are combined to form a new DataFrame.

#### Example:
- If we apply GroupBy on `Product_Name`, the data will be grouped based on the product name.

### `groupby()` Method

The `groupby()` method is a predefined function in the DataFrame class. It returns a `GroupBy` object, which can be used to perform various operations on the grouped data.

#### Syntax:
```python
grouped = df.groupby(["column_name"])  # Group by one or more columns
```

- **Parameters**:
  - The `groupby()` method accepts one or more columns to group by.

## Examples of GroupBy Operations

### 1. Grouping by a Single Column
Group the DataFrame by the `Product` column and calculate the sum for each group:
```python
grouped = df1.groupby(["Product"])
result = grouped.sum()
```

### 2. Grouping by a Column and Counting the Size of Each Group
Group the DataFrame by `Mail_Id` and get the size of each group:
```python
grouped = df1.groupby(["Mail_Id"])
result = grouped.size()
```

### 3. Grouping by Multiple Columns and Counting Values
Group by multiple columns (`Date`, `Product_Name`) and count the occurrences of each combination:
```python
cols = ['Date', 'Product_Name']
grouped = df1.groupby(cols)['Date']
result = grouped.count()
```

### 4. Grouping by Multiple Columns with `as_index=False`
Group the DataFrame by `Mail_Id` and `Product_Name` without setting them as the index, and calculate the sum of `Product_Cost`:
```python
col = ['Mail_Id', 'Product_Name']
grouped = df1.groupby(col, as_index=False)['Product_Cost']
result = grouped.sum()
```

---

## Summary

- **GroupBy**: A powerful operation for splitting, applying, and combining data in a DataFrame.
- **Steps**:
  1. **Splitting**: Divides data into groups based on some criteria.
  2. **Applying**: Applies a function to each group.
  3. **Combining**: Combines the results from all groups.
- **`groupby()`**: The method used to perform the grouping, which returns a `GroupBy` object that can be further used for various aggregation or transformation operations.

---


# Merging or Joining DataFrames

Merging or joining is a process of combining two DataFrames into one based on common attributes in columns. This operation is similar to the `JOIN` operation in databases. In pandas, the `merge()` function is used to perform this task.

## Introduction to Merging
- **Merging/Joining**: Combining two DataFrames based on shared columns.
- **Why**: It’s often necessary when we need to combine datasets that have related information.

## The `merge()` Function

The `merge()` function is a predefined function in pandas that allows us to perform various types of join operations between DataFrames.

### Syntax:
```python
pd.merge(df1, df2, on="column", how="join_type")
```
- **Parameters**:
  - `df1`: The first DataFrame.
  - `df2`: The second DataFrame.
  - `on`: The column(s) to join on.
  - `how`: The type of join (default is 'inner').

### The `how` Argument:
The `how` argument specifies the type of join to perform. Common join types include:
- **inner**: Keep only the rows with matching values in both DataFrames.
- **left**: Keep all rows from the left DataFrame and fill missing values from the right with `NaN`.
- **right**: Keep all rows from the right DataFrame and fill missing values from the left with `NaN`.
- **outer**: Keep all rows from both DataFrames, filling missing values with `NaN`.

## Types of Joins

### 1. Inner Join
- An **inner join** returns only the rows with matching data from both DataFrames.
- It’s like an intersection of the two DataFrames.
- **Syntax**:
```python
pd.merge(df1, df2, on="column", how="inner")
```

### 2. Left Join
- A **left join** keeps all rows from the left DataFrame and fills missing data from the right DataFrame with `NaN`.
- **Syntax**:
```python
pd.merge(df1, df2, on="column", how="left")
```

### 3. Right Join
- A **right join** keeps all rows from the right DataFrame and fills missing data from the left DataFrame with `NaN`.
- **Syntax**:
```python
pd.merge(df1, df2, on="column", how="right")
```

### 4. Outer Join (Full Outer Join)
- A **full outer join** returns all rows from both the left and right DataFrames. Missing values from either DataFrame are filled with `NaN`.
- **Syntax**:
```python
pd.merge(df1, df2, on="column", how="outer")
```

## Other Types of Joins

### 1. One-to-One Join
- A **one-to-one join** occurs when each row in the left DataFrame has a corresponding row in the right DataFrame.
- It’s similar to column-wise concatenation.
- **Syntax**:
```python
one_one = pd.merge(df1, df2)
```

### 2. Many-to-One Join
- A **many-to-one join** happens when one of the DataFrames has duplicate entries in the key column.
- **Syntax**:
```python
many_one = pd.merge(df1, df2)
```

### 3. Many-to-Many Join
- A **many-to-many join** happens when both DataFrames have duplicate entries in the key column.
- **Syntax**:
```python
many_many = pd.merge(df1, df2)
```

## Merging Based on Column(s)

You can merge DataFrames based on single or multiple columns.

- **Single Column**:
```python
result = pd.merge(df1, df2, on="Subject")
```

- **Multiple Columns**:
```python
result = pd.merge(df1, df2, on=["Name", "Subject"])
```

## Summary

- **Merging/Joining**: Combines two DataFrames based on common columns.
- **`merge()`**: The primary function for performing join operations.
- **Types of Joins**: 
  - **Inner**: Common data only.
  - **Left**: All left rows, missing values from the right filled with `NaN`.
  - **Right**: All right rows, missing values from the left filled with `NaN`.
  - **Outer**: All rows from both DataFrames, missing values filled with `NaN`.
- **Join Types**: 
  - **One-to-One**: One row in the left and right DataFrames match.
  - **Many-to-One**: Duplicate values in the left DataFrame.
  - **Many-to-Many**: Duplicate values in both DataFrames.

---


# Concatenating DataFrames

Concatenation in pandas allows us to combine or stack multiple DataFrames either vertically or horizontally based on specific requirements.

## The `concat()` Function

The `concat()` function in pandas is used to concatenate DataFrames along a particular axis (either rows or columns). This is a predefined function within the pandas library.

### Syntax:
```python
pd.concat([df1, df2], axis=0, ignore_index=False)
```

### Parameters:
- `df1, df2, ...`: DataFrames to be concatenated.
- `axis`: 
  - `0`: Concatenate along rows (default).
  - `1`: Concatenate along columns.
- `ignore_index`: 
  - `True`: Resets the index.
  - `False`: Retains the original index.

## Concatenation Examples

### 1. Concatenating Two DataFrames Vertically (along rows)
```python
result = [df1, df2]
df3 = pd.concat(result)
```
This combines the DataFrames vertically, stacking them one below the other.

### 2. Concatenating Two DataFrames with Reset Index
```python
df3 = pd.concat(result, ignore_index=True)
```
This resets the index, providing a continuous index for the resulting DataFrame.

### 3. Concatenating Two DataFrames Horizontally (along columns)
```python
df3 = pd.concat(result, axis=1)
```
This combines the DataFrames horizontally, aligning them side by side.

## Summary

- **`concat()`**: Concatenates DataFrames along rows (default) or columns.
- **`ignore_index=True`**: Resets the index in the resulting DataFrame.
- **`axis=1`**: Concatenates along columns, adding new columns to the DataFrame.

---


# Adding, Dropping Columns & Rows in DataFrame

In pandas, adding and removing columns or rows is a common operation that allows you to modify your DataFrame according to your needs.

## Adding a Column to DataFrame

### 1. Adding a New Column Based on Existing Columns
You can add a new column to a DataFrame by performing operations on existing columns.

```python
# Adding 'Total Cost' column by multiplying 'Product cost' and 'Quantity'
df = pd.read_csv("sales8.csv")
df["Total Cost"] = df['Product cost'] * df['Quantity']
```

### 2. Adding a Column Using the `apply()` Method
The `apply()` method can be used to apply a function row-wise to calculate the new column.

```python
# Function to calculate total cost
def total(df):
    return df['Product cost'] * df['Quantity']

# Adding 'Total cost' column
df['Total cost'] = df.apply(total, axis=1)
```

### 3. Adding a Column at a Specific Position
You can insert a column at a specific index position in the DataFrame using `insert()`.

```python
# Inserting 'Total Cost' column at the 5th position
new = df['Product cost'] * df['Quantity']
df.insert(5, "Total Cost", new)
```

## Dropping Columns from DataFrame

### 1. Dropping a Single Column
You can drop a single column from a DataFrame using the `drop()` method.

```python
# Dropping 'Customer name' column
df1 = pd.read_csv("sales8.csv")
df2 = df1.drop(columns='Customer name')
```

### 2. Dropping Multiple Columns
You can drop multiple columns at once by passing a list of column names.

```python
# Dropping 'Customer name' and 'Product name' columns
df1 = pd.read_csv("sales8.csv")
df2 = df1.drop(['Customer name', 'Product name'], axis=1)
```

## Dropping Rows from DataFrame

### 1. Dropping a Single Row
You can drop a row by specifying its index position using the `drop()` method.

```python
# Dropping row at index 3
df1 = pd.read_csv("sales8.csv")
df2 = df1.drop(3, axis=0)
```

### 2. Dropping Multiple Rows
You can drop multiple rows by passing a list of index positions.

```python
# Dropping rows at index positions 1 and 2
df1 = pd.read_csv("sales8.csv")
df2 = df1.drop([1, 2], axis=0)
```

---

## Summary

- **Adding Columns**: Add new columns either by performing operations on existing ones, using the `apply()` method, or inserting them at specific positions with `insert()`.
- **Dropping Columns**: Use the `drop()` method to remove one or more columns by specifying their names.
- **Dropping Rows**: Use the `drop()` method to remove rows by specifying their index positions.

---


# Date and Time Operations in Pandas

Date and time manipulations are crucial when working with datasets that contain time-based information. This guide covers how to handle date columns in pandas.

## Date Data Type in Pandas

When you load a CSV file containing date columns, pandas will treat these columns as objects (strings) by default. You can explicitly convert these columns into the `datetime` data type.

### Loading CSV with Date Columns

You can specify which columns should be parsed as dates when reading a CSV file using the `parse_dates` parameter.

```python
# Loading CSV file with date parsing
df = pd.read_csv('sales7_dates.csv', parse_dates=['Pur_Date'])
```

## Converting Object Data Type to Date Data Type

To convert an object data type (like a string) to a date data type, you can use one of two predefined pandas functions:

### 1. `to_datetime(p)`
The `to_datetime()` function is used to convert an object column to a date column.

```python
# Converting 'Pur_Date' from object to datetime
df = pd.read_csv('sales7_dates.csv')
df['Pur_Date'] = pd.to_datetime(df['Pur_Date'])
```

### 2. `astype(p)`
Alternatively, you can use the `astype()` function to convert the column to the `datetime64` type.

```python
# Converting 'Pur_Date' from object to datetime using astype
df['Pur_Date'] = df['Pur_Date'].astype('datetime64[ns]')
```

## Date Format Representation

Dates can be represented in various formats. Use the `format` parameter to specify the exact format for conversion.

```python
# Specifying the date format
df['PurDate'] = pd.to_datetime(df['PurDate'], format='%d%m%Y')
df['PurDate'] = pd.to_datetime(df['PurDate'], format='%d%b%Y')
```

## Handling Missing Dates (NaT)

If your date column contains missing values (`NaN`), converting them to datetime will raise an error. You can handle this by using the `errors="coerce"` argument, which converts invalid dates to `NaT` (Not a Time).

```python
# Converting dates with errors coerced to NaT
df['PurDate'] = pd.to_datetime(df['PurDate'], errors="coerce")
```

## Selecting Data Based on Date Range

You can filter data by selecting rows between specific start and end dates.

### Example: Selecting Data Between Dates

```python
# Selecting data between two dates
start = df['Pur_Date'] > '2019-01-01 01:00:00'
end = df['Pur_Date'] < '2019-01-01 05:00:00'
result = df[start & end]
```

## Accessing Specific Date Ranges

You can access data for specific time periods, such as the last 10 days, 1 month, or 1 year, by setting the date column as the index and using the `.last()` method.

### Examples:

```python
# Last 10 days of records
new_df = df.set_index("Pur_Date")
days_10 = new_df.last("10D")

# Last 40 days of records
days_40 = new_df.last("40D")

# Last 1 month of records
month_1 = new_df.last("1M")

# Last 1 year of records
year_1 = new_df.last("1Y")
```

## Extracting Components from Date

You can extract individual components (year, month, day, etc.) from the date column using pandas `dt` accessor.

### Example: Extracting Date Components

```python
# Extracting year, month, and day
df['year'] = df['Pur_Date'].dt.year
df['month'] = df['Pur_Date'].dt.month
df['day'] = df['Pur_Date'].dt.day
df['hour'] = df['Pur_Date'].dt.hour
df['minute'] = df['Pur_Date'].dt.minute
```

## Encoding Days of the Week

You can extract the day of the week for each date using the `day_name()` method or the `weekday` method. This is useful for analyzing trends like sales on specific weekdays.

### Example: Encoding Days of the Week

```python
# Getting the day of the week
df['PurDate'] = pd.to_datetime(df['PurDate'])
print(df["PurDate"].dt.day_name())  # Monday, Tuesday, etc.

# Getting the weekday number (Monday = 0, Sunday = 6)
print(df['PurDate'].dt.weekday)
```

---

## Summary of Date Operations in Pandas

- **Converting Object to Date**: Use `pd.to_datetime()` or `astype('datetime64[ns]')` to convert columns to date data types.
- **Handling Missing Dates**: Use `errors="coerce"` to handle missing or invalid dates.
- **Selecting Date Ranges**: Filter data between specific start and end dates or based on relative time periods (e.g., last 10 days, 1 month).
- **Extracting Date Components**: Use `.dt` accessor to extract year, month, day, hour, and minute from a date column.
- **Encoding Days of the Week**: Use `.dt.day_name()` or `.dt.weekday` to get the day of the week.

---


# Concatenating Multiple CSV Files in Python

In real-time scenarios, data is often spread across multiple files. For instance, sales data might be stored in monthly CSV files (e.g., January sales, February sales, etc.). To combine this data into one file, you can concatenate these CSV files.

## Real-World Use Case

When working with large datasets, data might be stored across multiple files (e.g., one file for each month). To generate yearly sales data, you need to concatenate these monthly files into one large file.

## Steps to Concatenate CSV Files

### 1. Access Files from a Folder

First, you need to access all files in a specific folder and filter out the CSV files.

#### Using the `os` Module

The `os` module is a built-in Python library that allows you to interact with the operating system. You can use it to list all files in a folder.

##### Example: Accessing Files from a Folder

```python
import os

path = "./daniel"  # Specify the folder path
all_files = os.listdir(path)  # List all files in the folder
```

### 2. Filter Only CSV Files

You can use the `filter()` function to filter out only the CSV files from the list.

#### Example: Filtering CSV Files

```python
# Filter only CSV files
f = filter(lambda name: name.endswith('.csv'), all_files)
csv_files = list(f)  # Convert the filter object to a list
```

### 3. Concatenate All CSV Files

Once you have all the CSV files, you can use pandas to load and concatenate them into one DataFrame.

#### Example: Concatenating CSV Files

```python
import glob
import pandas as pd

# Specify the folder path
p = './daniel'

# Get all CSV file paths
files = os.path.join(p, "*.csv")
csv_files = glob.glob(files)

# Read and concatenate all CSV files
result = (pd.read_csv(every) for every in csv_files)  # Generator expression
df = pd.concat(result, ignore_index=True)  # Concatenate all files into one DataFrame
```

### 4. Loading a CSV with Date Parsing

If your CSV files contain date columns, you can parse those dates while loading the data.

#### Example: Loading a CSV with Date Parsing

```python
# Load a CSV file with date parsing
df = pd.read_csv("year.csv", parse_dates=["Date"])
```

## Summary of Key Functions

- **`os.listdir()`**: Lists all files in a specified directory.
- **`filter()`**: Filters the files based on a condition (e.g., file extension).
- **`glob.glob()`**: Returns a list of file paths matching a specific pattern.
- **`pd.concat()`**: Concatenates multiple DataFrames into one.

By following these steps, you can easily concatenate multiple CSV files into one large DataFrame for analysis.



