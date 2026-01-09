
# Pandas Series

---

## **What is a Pandas Series?**
A **Series** is a 1-dimensional, labeled data structure in `pandas`, akin to a column in a spreadsheet or a single-dimension NumPy array with an index. It consists of:
- **Data**: Homogeneous elements (integers, floats, strings, etc.).
- **Index**: Labels for each element (default: 0-based integers, or custom labels).

Series are the building blocks of DataFrames (each DataFrame column is a Series), but they’re powerful on their own for handling ordered, labeled data.

---

## **1. Creating a Series**
You can create a Series from various sources: lists, dictionaries, NumPy arrays, scalars, etc.

### Getting a Series from a DataFrame
A DataFrame column is a Series. You can access it two ways:
- **Bracket Notation** (Preferred):
  ```python
  df = pd.read_csv("data.csv")
  df["column_name"]
  ```
- **Dot Notation** (Risky if names clash with methods):
  ```python
  df.column_name
  ```

---

## **Series Attributes**
- Series is a predefined class.
- Series having different attributes.
- These attributes return information about the object.

**Attributes**
Attributes give you information about a Series, such as its data, labels, and type.

| Attribute   | Description                           |
|-------------|---------------------------------------|
| `.values`    | Returns the data as a NumPy array.    |
| `.index`     | Shows the labels (e.g., 0, 1, 2...). |
| `.dtype`     | Tells the data type (e.g., `int64`).  |
| `.size`      | Counts the total number of items.     |
| `.name`      | Name of the Series (optional).     |

### **Example**
```python
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(s.values)    # Output: [10 20 30]
print(s.index)     # Output: Index(['a', 'b', 'c'], dtype='object')
print(s.dtype)     # Output: int64
print(s.size)      # Output: 3
s.name = "Numbers"
print(s)
# Output:
# a    10
# b    20
# c    30
# Name: Numbers, dtype: int64
```

---

## **Accessing and Modifying Data**
### **Indexing**
- **Label-based**: `s[label]`.
- **Position-based**: `s.iloc[integer]`.
- **Boolean indexing**: Filter with conditions.

### **Example**
```python
s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])

# Label-based
print(s["b"])      # Output: 20
print(s[["a", "c"]])  # Output: a    10 \n c    30

# Position-based
print(s.iloc[1])   # Output: 20
print(s.iloc[0:2]) # Output: a    10 \n b    20

# Boolean indexing
print(s[s > 25])   # Output: c    30 \n d    40
```

### **Modifying**
```python
# Updating value
s["a"] = 15
print(s)
# Output:
# a    15
# b    20
# c    30
# d    40

# Adding new element
s["e"] = 50
print(s)
# Output:
# a    15
# b    20
# c    30
# d    40
# e    50
```

---

## **Series Methods**

- Series is a predefined class.
- Series class having different methods
- These methods perform operations on Series of values.

- It is predefined method in Series class.
- We can access this method by using series object.

**Methods**
| Method       | Description                                |
|--------------|--------------------------------------------|
| `head(n)`    | Shows the first `n` items (default: 5).    |
| `tail(n)`    | Shows the last `n` items (default: 5).     |
| `sum()`      | Adds all values together.                  |
| `mean()`     | Calculates the average.                    |
| `describe()` | Gives stats like count, mean, and max.     |
| `unique()`   | Lists unique values.                       |
| `nunique()`  | Returns the number of unique values (ignores NaN by default).      |
| `count()`    | Returns the count of non-null values.  


```python
s = pd.Series([1, 2, 2, 3, 4])
print(s.head(3))  # Output: 0    1 \n 1    2 \n 2    2
print(s.describe())
# Output:
# count    5.000000
# mean     2.400000
# std      1.140175
# min      1.000000
# 25%      2.000000
# 50%      2.000000
# 75%      3.000000
# max      4.000000
print(s.value_counts())  # Output: 2    2 \n 1    1 \n 3    1 \n 4    1
```

### **Mathematical Operations**
- `.sum()`, `.mean()`, `.median()`, `.min()`, `.max()`, `.std()`, etc.

```python
print(s.sum())    # Output: 12
print(s.mean())   # Output: 2.4
print(s.max())    # Output: 4
```

### **Element-wise Operations**
```python
# Arithmetic
print(s * 2)  # Output: 0    2 \n 1    4 \n 2    4 \n 3    6 \n 4    8

# With another Series (aligned by index)
s2 = pd.Series([10, 20], index=[0, 1])
print(s + s2)
# Output:
# 0    11.0
# 1    22.0
# 2     NaN
# 3     NaN
# 4     NaN
```

### **Sorting**
- `.sort_values(ascending=True)`: Sort by values.
- `.sort_index()`: Sort by index.

```python
s = pd.Series([3, 1, 4], index=["b", "a", "c"])
print(s.sort_values())  # Output: a    1 \n b    3 \n c    4
print(s.sort_index())   # Output: a    1 \n b    3 \n c    4
```

### **Handling Missing Data**
- `.isna()`: Check for NaN.
- `.fillna(value)`: Replace NaN.
- `.dropna()`: Remove NaN.

```python
s = pd.Series([1, None, 3, None])
print(s.isna())    # Output: 0    False \n 1     True \n 2    False \n 3     True
print(s.fillna(0)) # Output: 0    1.0 \n 1    0.0 \n 2    3.0 \n 3    0.0
print(s.dropna())  # Output: 0    1.0 \n 2    3.0
```

---

## **Applying Functions**
- `.apply(func)`: Apply a function to each element.
- `.map(func)`: Map values to new values (often with a dictionary).

```python
s = pd.Series([1, 2, 3])
print(s.apply(lambda x: x**2))  # Output: 0    1 \n 1    4 \n 2    9

mapping = {1: "one", 2: "two", 3: "three"}
print(s.map(mapping))  # Output: 0     one \n 1     two \n 2    three
```

---

## **Tools and Utilities**
- **`.astype(type)`**: Convert data type.
  ```python
  s = pd.Series([1, 2, 3])
  print(s.astype(float))  # Output: 0    1.0 \n 1    2.0 \n 2    3.0
  ```
- **`.to_list()`**: Convert to Python list.
  ```python
  print(s.to_list())  # Output: [1, 2, 3]
  ```
- **`.to_frame()`**: Convert to DataFrame.
  ```python
  print(s.to_frame(name="Values"))
  # Output:
  #    Values
  # 0       1
  # 1       2
  # 2       3
  ```
- **`.idxmax()` / `.idxmin()`**: Index of max/min value.
  ```python
  print(s.idxmax())  # Output: 2
  ```

---

## **Advanced Features**
### **Datetime Handling**
```python
dates = pd.Series(pd.date_range("2023-01-01", periods=4, freq="M"))
print(dates)
# Output:
# 0   2023-01-31
# 1   2023-02-28
# 2   2023-03-31
# 3   2023-04-30
print(dates.dt.month)  # Extract month: 0    1 \n 1    2 \n 2    3 \n 3    4
```

### **Categorical Data**
```python
s = pd.Series(["low", "high", "medium"], dtype="category")
print(s)
# Output:
# 0       low
# 1      high
# 2    medium
# dtype: category
```

---

### **Conclusion**
The Pandas Series is a versatile, labeled 1D structure ideal for managing ordered data. 

