# 📚 Pandas: Interview Guide

## 1. Assessing DataFrame Emptiness 🎢

### Explanation:
- 🏷️ The `empty` attribute in a Pandas DataFrame is a built-in property.
- 🔧 It must be accessed via a DataFrame object.
- 🔄 This attribute returns `True` if the DataFrame contains no data; otherwise, it returns `False`.

```python
import pandas as pd

df = pd.DataFrame()

if df.empty:
    print('DataFrame is empty!')
```

---

## 2. Loading Data from External Files 💾

### Explanation:
- 📂 Utilize the `read_csv("file_path")` function to ingest CSV data into a DataFrame.

```python
import pandas as pd
df = pd.read_csv("sales1.csv")
print(df)
```

### Alternative File Formats and Corresponding Methods:
| 📚 File Type  | 🎯 Pandas Method |
|------------|--------------|
| 📄 CSV        | `pd.read_csv("file.csv")` |
| 📜 JSON       | `pd.read_json("file.json")` |
| 📊 Excel      | `pd.read_excel("file.xlsx")` |

### Handling Missing or Incorrect File Paths:
- 🚨 If the specified file is absent or the filename is incorrect, a `FileNotFoundError` is raised.

---

## 3. Retrieving DataFrame Dimensions and Column Labels 📈

### Explanation:
- 🔢 `df.shape` returns a tuple indicating the number of rows and columns.
- 🔧 `df.columns` provides a list of column names.

```python
import pandas as pd
df = pd.read_csv("sales1.csv")
print(df.shape)
print(df.columns)
```

---

## 4. DataFrame Overview via `info()` and `describe()` Methods 🎯

### `df.info()`
Provides comprehensive metadata, including:
- 🌍 Object type
- 📊 Column count and row count
- 📚 Column data types
- 🛠️ Memory footprint

### `df.describe()`
Generates a summary of key statistical metrics:
- 🌟 Count, mean, standard deviation
- 📈 Minimum, 25th, 50th, 75th percentile, and maximum values

```python
import pandas as pd

df = pd.read_csv("sales1.csv")
df.info()
df.describe()
```

---

## 5. Reordering DataFrame Columns 🗂

### Explanation:
- 🔄 Modify the sequence of DataFrame columns as required.

```python
import pandas as pd

df = pd.read_csv("sales1.csv")
df = df[["Product", "Customer Name", "Quantity", "Order ID"]]
```

---

## 6. Renaming DataFrame Columns 🔍

### Explanation:
- 📓 Use `rename()` with a dictionary to update column names efficiently.

```python
import pandas as pd
df = pd.read_csv("sales3.csv")

rename_dict = {
    'ord id': 'order_id',
    'cust name': 'customer_name',
    'cust id': 'customer_id',
    'prod name': 'product_name',
    'prod cost': 'product_cost'
}

df = df.rename(columns=rename_dict)
```

---

## 7. Understanding the `inplace` Parameter 🔄

### Explanation:
- 🗓 Setting `inplace=True` applies modifications directly to the original DataFrame, avoiding the creation of a new object.
- 💡 Commonly used in:
  - 📌 `rename()`
  - ✂️ `drop()`
  - 📊 `sort_values()`
  - 🔖 `set_index()`

```python
import pandas as pd
df = pd.read_csv("sales3.csv")
df.rename(columns=rename_dict, inplace=True)
```

---

## 8. Handling Missing Values (NaN) in Pandas 🌐

### Explanation:
- 🌟 **NaN (Not a Number)** denotes absent values within a DataFrame.
- 📚 The default datatype for NaN values is `float`.

```python
import pandas as pd
df = pd.read_csv("fruits1.csv")
```

---

## 9. Quantifying Missing Data in a DataFrame 🏢

### Explanation:
- 📊 Use `isna()` in combination with `sum()` to compute the number of missing values per column.

```python
import pandas as pd

df = pd.read_csv('fruits1.csv')
missing_values = df.isna().sum()

# Compute percentage of missing values
missing_percentage = (missing_values * 100) / len(df)
```

---

## 10. Eliminating Missing Values from a DataFrame 🗑

### Explanation:
- 🔒 The `dropna()` function removes rows containing NaN values.

```python
import pandas as pd
df = pd.read_csv("fruits1.csv")
df_cleaned = df.dropna()
```

---

# Pandas DataFrame Operations Guide

## 11. Converting Float Column to Integer Column

To convert a float column to an integer column in a Pandas DataFrame:

```python
import pandas as pd
df = pd.read_csv('fruits1.csv')
df_cleaned = df.dropna()  # Remove NaN values
df_int = df_cleaned.astype(int)  # Convert float columns to int
```

### Key Points:
- `astype(int)` is a predefined method in DataFrame.
- This method should be accessed via a DataFrame object.
- Converts float column values into integers.

---

## 12. Selecting Single and Multiple Columns

### Selecting a Single Column
Returns a **Series** when selecting a single column:

```python
import pandas as pd
df = pd.read_csv("sales1.csv")
print(df["Product"])
```

### Selecting Multiple Columns
Returns a **DataFrame** when selecting multiple columns:

```python
cols = ["Customer Name", "Product"]
df_selected = df[cols]
```

---

## 13. Selecting Specific Values from a Column

Use **Boolean conditions** to filter specific values:

```python
import pandas as pd
df = pd.read_csv("sales1.csv")
selected_rows = df[df["Product"] == "Macbook Pro Laptop"]
print(selected_rows)
```

---

## 14. `iloc` and `loc` in Pandas

Both are used as indexers to select rows and columns.

- **`iloc` (Integer-location based)**: Selects rows/columns using integer indices.
- **`loc` (Label-based)**: Selects rows/columns using labels.

### `iloc` - Integer-based Selection
```python
import pandas as pd
df = pd.read_csv('sales2.csv')
df_selected = df.iloc[0:5, 0:3]  # First 5 rows and first 3 columns
```

### `loc` - Label-based Selection
```python
import pandas as pd
df = pd.read_csv('sales2.csv')
filtered = df.loc[df['Customer name'] == 'Sagar', 'Product name':'Product cost']
print(filtered.head())
```

Selecting specific products with defined columns:

```python
df.set_index("Product name", inplace=True)
products = ['iPhone 9', 'ThinkPad Laptop']
columns = ['Product cost', 'Customer name']
df_selected = df.loc[products, columns]
print(df_selected)
```

---

## 17. Filtering Data in DataFrame

### Single Condition Filtering
```python
import pandas as pd
df = pd.read_csv("sales4.csv")
df_filtered = df[df['Product_Cost'] > 65000]
```

### Multiple Conditions Filtering
```python
condition1 = df['Product_Name'] == "iPhone 11"
condition2 = df['Customer_Name'] == "Nireekshan"
df_filtered = df[condition1 & condition2]
```

---

## 18. Sorting Columns in DataFrame

Sorting values using `sort_values()`:

```python
import pandas as pd
df = pd.read_csv("sales4.csv")
df_sorted = df.sort_values(by="Product_Cost")
```

Sorting in **descending order**:
```python
df_sorted = df.sort_values(by="Product_Cost", ascending=False)
```

Sorting by **Customer Name**:
```python
df_sorted = df.sort_values(by="Customer_Name")
```

---

## 19. Grouping Data with `groupby`

### Grouping by Product and Date
```python
import pandas as pd
df = pd.read_csv("sales5.csv")
grouped = df.groupby(["Product_Name", "Date"]).count()
```

### Applying Aggregations
```python
aggregations = {
    'Product_Cost': sum,
    'Product_Name': "count"
}
grouped = df.groupby(['Date', 'Product_Name']).agg(aggregations)
```

---

## 20. Join Operations in Pandas

Joining two DataFrames based on a common column.

### Types of Joins:
- **Inner Join**
- **Left Join (Left Outer Join)**
- **Right Join (Right Outer Join)**
- **Full Outer Join**

```python
import pandas as pd
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Score': [90, 80, 70]})

# Inner Join
result = pd.merge(df1, df2, on='ID', how='inner')
print(result)
```

---

# Pandas DataFrame Guide

## 21. Inner Join and Left Outer Join in Pandas

### Inner Join
- Only keeps rows where the merge "on" value exists in both the left and right DataFrames.

### Left Outer Join
- Keeps every row in the left DataFrame.
- Where there are missing values of the "on" variable in the right DataFrame, adds empty / NaN values in the result.

#### Code Example: Inner Join
```python
import pandas as pd

d1 = {
    "Id": [1, 2, 3, 4, 5, 6],
    "Name": ["Pradhan", "Venu", "Madhurima", "Nireekshan", "Shafi", "Veeru"],
    "Subject": ["English", "Java", "Html", "Python", "C", "dotnet"]
}

d2 = {
    "Id": [11, 12, 13, 14, 15, 16],
    "Name": ["Srinu", "Sumanth", "Neelima", "Daniel", "Arjun", "Veeru"],
    "Subject": ["Java", "Html", "Cpp", "Python", "C", "dotnet"]
}

df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)

inn_join = pd.merge(df1, df2, on="Subject", how="inner")
```

#### Code Example: Left Outer Join
```python
left_join = pd.merge(df1, df2, on="Subject", how="left")
```

---

## 22. Adding a Column to an Existing DataFrame
- We can add a new column to an existing DataFrame based on a calculation.

#### Code Example
```python
import pandas as pd 
df = pd.read_csv("sales8.csv")
df["Total Cost"] = df['Product cost'] * df['Quantity']
print(df.head(5))
```

---

## 23. Deleting a Column from an Existing DataFrame
- Columns can be dropped using the `drop()` method.

#### Code Example: Dropping a Single Column
```python
import pandas as pd 
df1 = pd.read_csv("sales8.csv")
df2 = df1.drop(columns=['Customer name'])
```

---

## 24. Loading a CSV File with a Date Column
- The `parse_dates` argument should be used to correctly interpret date columns.

#### Code Example
```python
import pandas as pd 
df = pd.read_csv('sales7_dates.csv', parse_dates=['Pur_Date'])
```

---

## 25. Understanding NaT Values
- If a Date column contains missing values, converting it directly will cause an error.
- Use `errors="coerce"` to handle missing values as `NaT` (Not a Time).

#### Code Example: Handling Missing Dates
```python
import pandas as pd 

data = {
    "Product": ["Samsung", "iPhone", "Motorola"],
    "Status": ["Success", "Success", "Failed"],
    "Cost": [10000, 50000, 15000],
    "PurDate": ['02-Sep-2019', 'Here date is missing', '21-Sep-2019']
}

df = pd.DataFrame(data)
df['PurDate'] = pd.to_datetime(df['PurDate'], errors="coerce")
```

---

## 26. Extracting Year, Month, and Day from a Date Column
- Useful for breaking date columns into separate features.

#### Code Example
```python
import pandas as pd 
df = pd.read_csv('sales7_dates.csv', parse_dates=['Pur_Date'])
df['year'] = df['Pur_Date'].dt.year
df['month'] = df['Pur_Date'].dt.month
df['day'] = df['Pur_Date'].dt.day
print(df.head())
```

---

## 27. Concatenating Multiple CSV Files into One
- Often, data is stored across multiple files (e.g., monthly sales reports). 
- Concatenation helps merge these files into a single dataset.

#### Code Example
```python
import os
import glob
import pandas as pd

p = './daniel'
files = os.path.join(p, "*.csv")
csv_files = glob.glob(files)

result = (pd.read_csv(every) for every in csv_files)
df = pd.concat(result, ignore_index=True)
print(df)
df.to_csv("year.csv", index=False)
```

---