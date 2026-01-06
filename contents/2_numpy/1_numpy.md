
# NumPy (**Numerical Python**)

---

# Fundamentals of NumPy

NumPy provides powerful tools for handling arrays and matrices, making it ideal for numerical computations.

---

## What is NumPy?

- NumPy (**Numerical Python**) is a **fundamental library** for scientific computing in Python. 
- It provides an **efficient multi-dimensional array object (`ndarray`)** and a collection of mathematical functions to operate on these arrays efficiently.
- It’s the backbone of many data science libraries like `pandas`, `scikit-learn`, and `tensorflow`.

---

## Why Use NumPy?

- **ndarray**: Fast, memory-efficient multi-dimensional array object.
- **Vectorized Operations**: Perform computations without explicit loops.
- **Broadcasting**: Operate on arrays of different shapes seamlessly.
- **Mathematical Functions**: Extensive toolkit for linear algebra, statistics, and more.

---

## What is an Array?

An **array** is an **ordered collection** of values where all elements have the **same data type**. NumPy provides **ndarrays**, which allow fast operations on large datasets.

💡 **Key Concept**: The core array object in NumPy is called `ndarray`.

---

## Creating a NumPy Array

- We can create a **NumPy array** using the `np.array()` function.
- NumPy internally creates an **ndarray (n-dimensional array)** object.
- We can pass **lists, tuples, etc.** as parameters to the `array()` function.
- Using **values of the same type** in an array is recommended for efficiency.

### Creating a NumPy Array with a Single Value
```python
import numpy as np
age = 55
value = np.array(age)
```

### Creating a NumPy Array with Multiple Values
```python
import numpy as np
details = [10, 20, 30, 40, 50]
sales = np.array(details)
```

### Creating a 2D NumPy Array (Matrix)
```python
import numpy as np
details = [[10, 20], [30, 40]]
sales = np.array(details)
```

---

## Checking Array Dimensions

- `ndim` is a **predefined attribute** in NumPy.
- It helps us **check the dimensions** of an array.

```python
import numpy as np
array_1d = np.array([1, 2, 3])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

print(array_1d.ndim)  # Output: 1
print(array_2d.ndim)  # Output: 2
```

---

## Creating Arrays with Default Values

**Using Built-in Functions**
- `np.zeros()`: Array of zeros.
- `np.ones()`: Array of ones.
- `np.arange()`: Array with a range of values.
- `np.linspace()`: Evenly spaced values.

```python
# Zeros
zeros = np.zeros((2, 3))
print(zeros)
# Output:
# [[0. 0. 0.]
#  [0. 0. 0.]]

# Ones
ones = np.ones((3, 2))
print(ones)
# Output:
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]

# Range
range_arr = np.arange(0, 10, 2)  # Start, stop, step
print(range_arr)  # Output: [0 2 4 6 8]

# Linearly spaced
linspace = np.linspace(0, 1, 5)  # Start, stop, num
print(linspace)  # Output: [0.   0.25 0.5  0.75 1.  ]
```

---
## **Random Arrays**
- `np.random.rand()`: Uniform random numbers (0 to 1).
- `np.random.randn()`: Standard normal distribution.
- `np.random.randint()`: Random integers.

```python
# Uniform random
rand_arr = np.random.rand(2, 3)
print(rand_arr)
# Output: e.g.,
# [[0.5488135  0.71518937 0.60276338]
#  [0.54488318 0.4236548  0.64589411]]

# Normal random
norm_arr = np.random.randn(2, 2)
print(norm_arr)
# Output: e.g.,
# [[ 1.76405235  0.40015721]
#  [ 0.97873798  2.2408932 ]]

# Random integers
int_arr = np.random.randint(0, 10, size=(3, 2))
print(int_arr)
# Output: e.g.,
# [[4 7]
#  [2 9]
#  [5 1]]
```

---

# 📌 **NumPy Array Attributes**  

NumPy arrays come with **predefined attributes** that help us understand their **structure and functionality**.  

---

## 🔍 **NumPy Array Attributes**  
- `shape` attribute **returns a tuple** representing the **number of rows and columns** in an array.  
- `ndim` attribute **returns the number of dimensions (1D, 2D)** of an array. 
- `T` attribute **transposes an array**, converting **rows into columns** and **columns into rows**.  
- `size` attribute Total number of elements.

```python
import numpy as np

details = np.array([[10, 20, 30], [40, 50, 60]])
sales = np.array(details)

print(sales.shape)  # Output: (2, 3) → 2 rows, 3 columns
print(sales.ndim)  # Output: 1 → 1D array
print(sales.T)
""" [[10 40]
    [20 50]
    [30 60]] """
```
💡 **Notice** how the **rows and columns are swapped**.

### **Reshaping**
- `.reshape()`: Changes the **structure** of an array without changing its data.
- `.flatten()`: Converts a multi-dimensional array into a **one-dimensional** array.

```python
arr = np.arange(6)  # [0 1 2 3 4 5]
reshaped = arr.reshape(2, 3)
print(reshaped)
# Output:
# [[0 1 2]
#  [3 4 5]]

flattened = reshaped.flatten()
print(flattened)  # Output: [0 1 2 3 4 5]
```

## Indexing and Slicing in NumPy

- NumPy arrays **follow zero-based indexing**:
  - **First element** → stored at **index 0**
  - **Second element** → stored at **index 1**, and so on.
- **Slicing** allows extracting a portion of the array.

### Accessing Elements Using Indexing
```python
import numpy as np
details = [10, 20, 30, 40, 50]
sales = np.array(details)

print(sales[0])  # Access first element
```

### Accessing Elements Using Slicing
```python
import numpy as np
details = [10, 20, 30, 40, 50]
sales = np.array(details)

print(sales[2:])  # Access elements from index 2 onward
```

### Creating a Matrix and Selecting Elements
```python
import numpy as np
matrix = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print(matrix)          # Print the matrix
print(matrix[0,0])     # Access element at row 0, column 0
print(matrix[1,1])     # Access element at row 1, column 1
print(matrix[2,2])     # Access element at row 2, column 2
```

---

## Handling Index Errors

- If we **access an index out of bounds**, we get an `IndexError`.

### Example of Index Error
```python
import numpy as np
details = [10, 20, 30, 40, 50]
sales = np.array(details)

print(sales[22])  # Trying to access an out-of-bounds index
```

🔴 **Error Message:**
```
IndexError: index 22 is out of bounds for axis 0 with size 5
```

---

# 📌 **NumPy Array Methods**  

NumPy provides **predefined methods** to perform various operations on arrays efficiently.  

---

## **Mathematical Operations**
NumPy excels at vectorized operations, avoiding Python loops for efficiency.

### **1. Element-wise Operations**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Arithmetic
print(a + b)  # np.add(a, b) Output: [5 7 9]
print(a * b)  # Output: [ 4 10 18]
print(a - 2)  # np.subtract(a, b) Output: [-1  0  1]

# Universal functions (ufuncs)
print(np.sqrt(a))  # Output: [1.         1.41421356 1.73205081]
print(np.exp(b))   # Output: [ 54.59815003 148.4131591  403.42879349]
```
### **2. Broadcasting**
Operate on arrays of different shapes.
```python
arr = np.array([[1, 2], [3, 4]])
scalar = 2
print(arr * scalar)
# Output:
# [[2 4]
#  [6 8]]

row_vec = np.array([1, 2])
print(arr + row_vec)
# Output:
# [[2 4]
#  [4 6]]
```

### **3. Aggregation Functions**
- `sum()`, `mean()`, `std()`, `min()`, `max()`, etc.
- `min()`: Returns the **smallest value** in the array.
- `max()`: Returns the **largest value** in the array. 
- `sum()`: Returns the **sum of all elements** in the array.  

```python
arr = np.array([[1, 2], [3, 4]])
print(np.sum(arr))       # Output: 10
print(np.mean(arr))      # Output: 2.5
print(np.std(arr))       # Output: 1.118033988749895
print(np.max(arr, axis=0))  # Output: [3 4] (column-wise max)
```

---
## **Linear Algebra**
NumPy provides tools for matrix operations.

### **Dot Product**
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Matrix multiplication
result = np.dot(a, b)
print(result)
# Output:
# [[19 22]
#  [43 50]]
```

### **Trace**
- `trace()`: Calculates the **sum of diagonal elements**.
```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Trace of the Matrix:", matrix.trace())  
print("Sum of Diagonal Elements:", sum(matrix.diagonal()))
```

### **Transpose**
```python
print(a.T)
# Output:
# [[1 3]
#  [2 4]]
```

### **Inverse and Determinant**
```python
from numpy.linalg import inv, det

print(inv(a))
# Output: e.g.,
# [[-2.   1. ]
#  [ 1.5 -0.5]]

print(det(a))  # Output: -2.0000000000000004
```

### **Eigenvalues and Eigenvectors**
```python
from numpy.linalg import eig

eigenvalues, eigenvectors = eig(a)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
# Output: e.g.,
# Eigenvalues: [-0.37228132  5.37228132]
# Eigenvectors:
# [[-0.82456484 -0.41597356]
#  [ 0.56576746 -0.90937671]]
```

---

## **Statistical Functions**
NumPy supports statistical analysis.

### **Basic Statistics**
```python
data = np.array([1, 2, 3, 4, 5])
print(np.median(data))  # Output: 3.0
print(np.percentile(data, 75))  # Output: 4.0
print(np.var(data))  # Output: 2.0
```

### **Random Sampling**
```python
# Sample from normal distribution
sample = np.random.normal(loc=0, scale=1, size=10)
print(sample)  # Output: Random array, e.g., [-0.12  0.45 ...]
```

---

## **Array Manipulation**
### **Concatenation**
```python
a = np.array([1, 2])
b = np.array([3, 4])
print(np.concatenate((a, b)))  # Output: [1 2 3 4]

# Stack vertically
c = np.vstack((a, b))
print(c)
# Output:
# [[1 2]
#  [3 4]]
```

### **Splitting**
```python
arr = np.array([1, 2, 3, 4, 5, 6])
split_arr = np.split(arr, 3)
print(split_arr)  # Output: [array([1, 2]), array([3, 4]), array([5, 6])]
```

### **Sorting**
- `sort()`: Sorts the values in ascending order.

```python
arr = np.array([3, 1, 4, 2])
print(np.sort(arr))  # Output: [1 2 3 4]
print(np.argsort(arr))  # Output: [1 3 0 2] (indices of sorted order)
```

---
## **Tools and Methods Summary**
- **Creation**: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`, `np.random.*`.
- **Attributes**: `.shape`, `.ndim`, `.size`, `.dtype`.
- **Manipulation**: `.reshape()`, `.flatten()`, `np.vstack()`, `np.split()`, `np.sort()`.
- **Math Ops**: `+`, `*`, `np.dot()`, `np.sqrt()`, `np.exp()`, `np.sum()`, `np.mean()`.
- **Linear Algebra**: `np.linalg.inv()`, `np.linalg.det()`, `np.linalg.eig()`.
- **Statistics**: `np.median()`, `np.var()`, `np.corrcoef()`.

---