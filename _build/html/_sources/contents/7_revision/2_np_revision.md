
# NumPy in Python

NumPy is a fundamental library for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

## What is NumPy?

- **Full form**: NumPy stands for **Numerical Python**.
- **Creator**: It was created by Travis Oliphant.
- **Core Functionality**: NumPy provides the `ndarray` (n-dimensional array) object to store and manipulate data in the form of arrays.

### Key Features of NumPy:

- **Multidimensional Arrays**: NumPy arrays allow you to work with large datasets efficiently.
- **Faster than Python Lists**: NumPy arrays are more efficient and faster than Python lists for numeric operations.
- **Open Source**: NumPy is free to use and distributed under an open-source license.

## What is an Array?

- An **array** is an object that stores a collection of values.
- It is also called an **ordered collection** of values, where all the elements are of the same data type (e.g., all integers, all floats, etc.).

## Why Use NumPy Arrays?

- **Performance**: NumPy arrays are faster and more efficient than Python lists for numerical operations.
- **Memory Efficiency**: NumPy arrays are stored more compactly, allowing for faster access and manipulation.

The core object in NumPy is called the `ndarray` (n-dimensional array), which allows for operations on large datasets with minimal memory overhead.

## Installing NumPy

NumPy is not part of the default Python installation, so you will need to install it explicitly.

### Installation Steps:

1. Open your command prompt or terminal.
2. Run the following command to install NumPy using `pip`:

```bash
pip install numpy
```

### What is `pip`?

- `pip` stands for **Python Installer Package** and is a package management system used to install and manage software packages in Python.
- To install NumPy or any other Python package, you use the following command:

```bash
pip install package_name
```

## Importing NumPy

Once NumPy is installed, you can import it into your Python code.

### Import Syntax:

```python
import numpy
```

However, it’s common to give NumPy an alias for convenience, typically `np`.

### Using Alias for NumPy:

```python
import numpy as np
```

Using an alias makes it easier to reference NumPy functions without having to type `numpy` each time.

## Handling Import Errors

If NumPy is not installed properly, you may encounter an error like:

```
ModuleNotFoundError: No module named 'numpy'
```

To resolve this, ensure that you have installed NumPy correctly using `pip install numpy`.

---

### Summary

- **NumPy** is a library for numerical operations in Python.
- It provides **ndarray**, a fast and efficient array object.
- NumPy is **open source** and must be installed via `pip`.
- You can use **`import numpy as np`** for convenience when using NumPy in your code.

---


# Fundamentals of NumPy Arrays

NumPy provides powerful tools for handling arrays and matrices, making it ideal for numerical computations. Below are some of the basic operations in NumPy that help you get started.

## Creating a NumPy Array

You can create NumPy arrays using the `np.array()` function. NumPy arrays are powerful because they allow efficient storage and manipulation of numerical data.

### Syntax:
```python
import numpy as np
np.array(data)
```

- You can pass lists, tuples, or other sequences as arguments to create an array.
- It is recommended that all values in a NumPy array be of the same type for better performance.

### Example 1: Creating a NumPy Array from a Single Value
```python
import numpy as np
age = 44
value = np.array(age)
```

### Example 2: Creating a NumPy Array from a List
```python
import numpy as np
details = [10, 20, 30, 40, 50]
sales = np.array(details)
```

### Example 3: Creating a 2D NumPy Array (Matrix)
```python
import numpy as np
details = [[10, 20], [30, 40]]
sales = np.array(details)
```

## Array Dimensions (`ndim`)

The `ndim` attribute is used to check the number of dimensions in an array.

### Example:
```python
import numpy as np
array = np.array([10, 20, 30])
print(array.ndim)  # Output: 1 (1D array)

matrix = np.array([[1, 2], [3, 4]])
print(matrix.ndim)  # Output: 2 (2D array)
```

## Indexing and Slicing NumPy Arrays

### Indexing:
NumPy arrays are indexed, with the first element being at index `0`, the second at index `1`, and so on.

### Example 1: Accessing an Element by Index
```python
import numpy as np
details = [10, 20, 30, 40, 50]
sales = np.array(details)

print(sales[0])  # Output: 10
```

### Example 2: Slicing the Array (Accessing a Subset of Elements)
```python
import numpy as np
details = [10, 20, 30, 40, 50]
sales = np.array(details)

print(sales[2:])  # Output: [30, 40, 50]
```

### Example 3: Accessing Elements in a 2D Array (Matrix)
```python
import numpy as np
matrix = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

print(matrix[0, 0])  # Output: 10
print(matrix[1, 1])  # Output: 50
print(matrix[2, 2])  # Output: 90
```

## IndexError in NumPy

If you try to access an index that is out of bounds, NumPy will raise an `IndexError`.

### Example:
```python
import numpy as np
details = [10, 20, 30, 40, 50]
sales = np.array(details)

print(sales[22])  # IndexError: index 22 is out of bounds for axis 0 with size 5
```

## Creating Arrays with Specific Values

### Creating an Array of Zeros:
To create an array filled with zeros, use `np.zeros()`.

### Example:
```python
import numpy as np
sales = np.zeros(5)  # Output: [0. 0. 0. 0. 0.]
```

### Creating an Array of Ones:
To create an array filled with ones, use `np.ones()`.

### Example:
```python
import numpy as np
sales = np.ones(5)  # Output: [1. 1. 1. 1. 1.]
```

---

### Summary:
- **Creating Arrays**: Use `np.array()` to create arrays from lists, tuples, etc.
- **Array Dimensions**: Use `.ndim` to check the number of dimensions of an array.
- **Indexing and Slicing**: You can access array elements using indexing and slicing.
- **Handling Index Errors**: Accessing out-of-bounds indices raises an `IndexError`.
- **Creating Specialized Arrays**: Use `np.zeros()` and `np.ones()` to create arrays filled with zeros or ones.

---


# NumPy Array Attributes

NumPy arrays come with several predefined attributes that allow you to understand and manipulate their structure efficiently. Below are some essential attributes of NumPy arrays.

## 1. `shape` Attribute

The `shape` attribute provides information about the dimensions of the array, such as the number of rows and columns. It returns a tuple representing the array's dimensions.

### Example:
```python
import numpy as np
details = np.array([[10, 20, 30], [40, 50, 60]])
sales = np.array(details)

print(sales.shape)  # Output: (2, 3) (2 rows, 3 columns)
```

## 2. `ndim` Attribute

The `ndim` attribute tells you the number of dimensions in the array. It is useful for understanding the structure of the data.

### Example:
```python
import numpy as np
details = [10, 20, 30, 40, 50]
sales = np.array(details)

print(sales.ndim)  # Output: 1 (1-dimensional array)
```

## 3. `T` Attribute (Transpose)

The `T` attribute is used to transpose the array. It swaps the rows and columns, turning rows into columns and columns into rows.

### Example:
```python
import numpy as np
details = [[10, 20, 30], [40, 50, 60]]
sales = np.array(details)

print(sales.T)
# Output:
# [[10 40]
#  [20 50]
#  [30 60]]
```

---

### Summary of Common NumPy Array Attributes:
- **`shape`**: Returns the number of rows and columns (tuple).
- **`ndim`**: Returns the number of dimensions of the array.
- **`T`**: Transposes the array (swaps rows with columns).

---


# NumPy Array Methods

NumPy arrays come with several predefined methods that help perform various operations on the array. Below are some common methods and their use cases.

## 1. `min()` Method
The `min()` method returns the minimum value from the array.

### Example:
```python
import numpy as np
details = np.array([[10, 20, 30], [40, 50, 60]])
print(details.min())  # Output: 10
```

## 2. `max()` Method
The `max()` method returns the maximum value from the array.

### Example:
```python
import numpy as np
details = np.array([[10, 20, 30], [40, 50, 60]])
print(details.max())  # Output: 60
```

## 3. `sum()` Method
The `sum()` method returns the sum of all values in the array.

### Example:
```python
import numpy as np
details = np.array([[10, 20, 30], [40, 50, 60]])
print(details.sum())  # Output: 210
```

## 4. `reshape()` Method
The `reshape()` method changes the shape of an array without changing its data.

### Example:
```python
import numpy as np
details = np.array([[10, 20, 30], [40, 50, 60]])
print(details.reshape(3, 2))  # Output: reshaped array with 3 rows and 2 columns
```

## 5. `sort()` Method
The `sort()` method sorts the values in the array.

### Example:
```python
import numpy as np
details = np.array([[55, 13, 12], [99, 2, 1]])
details.sort()
print(details)  # Output: sorted array
```

## 6. `flatten()` Method
The `flatten()` method converts a multi-dimensional array into a 1-dimensional array.

### Example:
```python
import numpy as np
details = np.array([[10, 20, 30], [40, 50, 60]])
print(details.flatten())  # Output: [10 20 30 40 50 60]
```

## 7. `diagonal()` Method
The `diagonal()` method returns the diagonal elements of a matrix.

### Example:
```python
import numpy as np
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix.diagonal())  # Output: [1 5 9]
```

## 8. `trace()` Method
The `trace()` method returns the sum of the diagonal elements of a matrix.

### Example:
```python
import numpy as np
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix.trace())  # Output: 15 (sum of 1 + 5 + 9)
```

## 9. `count_nonzero()` Method
The `count_nonzero()` method counts the number of non-zero values in the array.

### Example:
```python
import numpy as np
details = np.array([[10, 0, 30], [40, 50, 0]])
print(np.count_nonzero(details))  # Output: 4
```

## 10. Adding a Value to an Array
You can add a constant value to all elements of a NumPy array.

### Example:
```python
import numpy as np
details = np.array([[10, 20, 30], [40, 50, 60]])
print(details + 2)  # Output: Adds 2 to each element of the array
```

## 11. Adding and Subtracting Matrices
You can add and subtract matrices using `np.add()` and `np.subtract()` methods.

### Adding Two Matrices:
```python
import numpy as np
matrix_a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 2]])
matrix_b = np.array([[1, 3, 1], [1, 3, 1], [1, 3, 8]])

print(np.add(matrix_a, matrix_b))  # Output: Element-wise addition
```

### Subtracting Two Matrices:
```python
import numpy as np
matrix_a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 2]])
matrix_b = np.array([[1, 3, 1], [1, 3, 1], [1, 3, 8]])

print(np.subtract(matrix_a, matrix_b))  # Output: Element-wise subtraction
```

### Adding and Subtracting Matrices Using `+` and `-` Operators:
```python
import numpy as np
matrix_a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 2]])
matrix_b = np.array([[1, 3, 1], [1, 3, 1], [1, 3, 8]])

print(matrix_a + matrix_b)  # Output: Element-wise addition using '+'
print(matrix_a - matrix_b)  # Output: Element-wise subtraction using '-'
```

---

### Summary of Common NumPy Array Methods:
- **`min()`**: Returns the minimum value in the array.
- **`max()`**: Returns the maximum value in the array.
- **`sum()`**: Returns the sum of all elements in the array.
- **`reshape()`**: Changes the shape of the array.
- **`sort()`**: Sorts the array in ascending order.
- **`flatten()`**: Flattens the array into a 1D array.
- **`diagonal()`**: Returns the diagonal elements of a matrix.
- **`trace()`**: Returns the sum of the diagonal elements of a matrix.
- **`count_nonzero()`**: Counts the number of non-zero elements in the array.
- **Array Operations**: You can add, subtract, and manipulate arrays using simple operators and predefined functions like `np.add()` and `np.subtract()`.

---

