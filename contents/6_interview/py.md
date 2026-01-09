# Python: Interview Guide

---
## 1. Python Libraries in a Project

Below are some commonly used Python libraries in projects:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `joblib`
- `matplotlib`
- `beautifulsoup4`
- `scrapy`

---

## 2. Installing Python Libraries

Python libraries can be installed using `pip`:

```bash
pip install numpy
pip install pandas
```

---

## 3. What is PIP in Python?

- PIP stands for **Python Installer Package**.
- It is a package management system used to install and manage Python libraries.
- Example usage:

```bash
pip install <package_name>
```

---

## 4. Requirements File (`requirements.txt`)

- A `requirements.txt` file helps install multiple libraries at once.
- To install all required libraries:

```bash
pip install -r requirements.txt
```

---

# Basics of Python

## 5. How Python Executes Code

1. The Python **interpreter** converts Python code into **bytecode**.
2. The **Python Virtual Machine (PVM)** converts bytecode into a machine-readable format.
3. The final output is displayed.

---

## 6. Built-in Data Types in Python

Python supports various data types:

- **Numbers:** `int`, `float`, `complex`
- **Sequences:** `str`, `list`, `set`, `tuple`, `dict`, `range`
- **Others:** `None`, `bool`

---

## 7. What is `None` in Python?

- `None` is a **special keyword** that represents the absence of a value.
- Functions without a return statement **implicitly return `None`**.

Example:

```python
def example():
    pass

print(example())  # Output: None
```

---

## 8. `for` Loop in Python

The `for` loop is used to iterate over a sequence:

```python
def get_info(sales):
    for sale in sales:
        print(sale)

data = [100, 200, 309, 440]
get_info(data)
```

---

# Data Structure

---

## 9. Mutable vs Immutable Objects

- **Mutable**: Can be changed after creation.
  - Examples: `list`, `dictionary`, `set`
- **Immutable**: Cannot be changed after creation.
  - Examples: `int`, `float`, `str`, `tuple`, `range`

#### Mutable Example (List):
```python
def modify_list(lst):
    print("Before modifying:", lst)
    lst[1] = 999
    print("After modifying:", lst)

a = [10, 20, 30, 40]
modify_list(a)
```

#### Immutable Example (Tuple):
```python
def modify_tuple(t):
    print("Before modifying:", t)
    t[1] = 999  # This will raise an error
    print("After modifying:", t)

t = (10, 20, 30, 40)
modify_tuple(t)
```
**Error:** `TypeError: 'tuple' object does not support item assignment`

---

## 10. `replace(p1, p2)` Method in Strings

- `replace(p1, p2)` replaces an **old substring (`p1`) with a new substring (`p2`)**.
- **Does not modify** the original string, creates a new one.

Example:

```python
old_name = "Prasaaaaad Kumar"
updated = old_name.replace("Prasaaaaad", "Prasad")

print("Name in Aadhar Card:", old_name)  # Original remains unchanged
print("Updated name in Aadhar Card:", updated)  # New string created
print("Name id is:", id(old_name))
print("Updated name id is:", id(updated))
```

---

# Python Interview Guide for Developers

## 11. `split(p)` and `join(p)` in Python

`split(p)` and `join(p)` are predefined string methods.

- `split(p)`: Splits a string based on a delimiter (space, character, or symbol) and returns a list of strings.
- `join(p)`: Joins a list of strings using a delimiter and returns a single string.

### Example:
```python
# Splitting a string
name = "This is Daniel"
print(name.split(" "))

# Joining a list of strings
details = ["Daniel", "Data Scientist", "Google"]
print(" ".join(details))
```

---

## 12. How to Reverse a String?
Using slicing, we can reverse a string.

### Example:
```python
name = "Daniel"
print(name[::-1])
```

---

# Function

---

## 13. `return` Keyword in Python
- `return` is a keyword used in functions/methods to return a value.

### Example:
```python
# Function that returns the maximum value from a list
def max_value(p):
    result = max(p)
    return result

values = [10, 20, 30, 40, 50]
print(max_value(values))
```

---

## 14. Can a Function Return Multiple Values?
Yes, Python functions can return multiple values as a tuple.

### Example:
```python
def details(name, qualification):
    return name, qualification

n, q = details("Daniel", "MCA")
print(n, q)
```

---

## 15. `map(fun, iterable)` Function
- `map(fun, iterable)` applies a function to each item in an iterable and returns a new object.
- Convert it to `list()` to see the result.

### Example:
```python
no_gst = [100, 200, 300, 400]
with_gst_cost = map(lambda item: item + 10, no_gst)

print(list(with_gst_cost))
```

---

## 16. `filter(fun, iterable)` Function
- `filter(fun, iterable)` applies a filtering function and returns only matching items.
- Convert it to `list()` to see the result.

### Example:
```python
sales = [100, 200, 300, 400, 500, 600, 700, 800]
ls_600 = filter(lambda sale: sale < 600, sales)

print(list(ls_600))
```

---

## 17. `reduce(fun, iterable)` Function
- `reduce(fun, iterable)` applies a function cumulatively to items in a sequence.
- Available in `functools` module.

### Example:
```python
from functools import reduce

sales = [100, 200, 300, 400, 500, 600, 700, 800]
total_sales = reduce(lambda x, y: x + y, sales)

print("Total sales:", total_sales)
```

---

## 18. Keyword Arguments
- We can pass values to function parameters using named identifiers.

### Example:
```python
def details(name, age):
    print("Student Name:", name)
    print("Student Age:", age)

details(name="Prasad", age=16)
```

---

## 19. Default Arguments
- We can provide default values for function parameters.

### Example:
```python
def product(name, price=100):
    print("Product Name:", name)
    print("Product cost:", price)

product(name="Bucket")
```

---

## 20. `*args` (Variable Length Arguments)
- Used when a function needs to accept a variable number of arguments.
- Values are stored as a tuple internally.

### Example:
```python
def total_sales(*items):
    return sum(items)

print("Total cost:", total_sales(250, 200, 300, 100))
```

---

## 21. **`**kwargs` (Keyword Variable Length Arguments)**

### What is `**kwargs`?
- `**kwargs` allows us to pass a variable number of keyword arguments to a function.
- It uses a double asterisk (`**`) before the parameter name.
- Internally, it stores key-value pairs in a dictionary.
- Applicable to:
  - Functions
  - Methods
  - Constructors

```python
# Example: Using **kwargs

def total_sales(**sales):
    print("Total sales:", sales)

total_sales(item1=250, item2=200, item3=300)
```

---

## 22. **Lambda (Anonymous Function)**

### What is a Lambda Function?
- A function without a name is called an **anonymous function**.
- We use the `lambda` keyword to define it.
- It processes inputs and returns a result in a single expression.

```python
# Example: Lambda function to square a number
square_value = lambda num: num * num
result = square_value(5)
print("Square value is:", result)
```

---

## 23. **Decorators**

### What are Decorators?
- Functions in Python are first-class objects, meaning:
  - A function can be assigned to a variable.
  - A function can be passed as an argument.
  - A function can return another function.
- A **decorator** is a function that takes another function as an argument and extends its behavior without modifying it.

---

## 24. **Generators (`yield` keyword)**

### What are Generators?
- A function with `yield` is a **generator**.
- Generators return **iterators** and generate values **on the fly**.
- They are memory-efficient.

```python
# Example: Generator function

def gen(n):
    yield n
    yield n + 1
 
g = gen(6)
print(next(g))
print(next(g))
```

---

## 25. **Monkey Patching**

### What is Monkey Patching?
- A technique to **dynamically modify the behavior** of code **at runtime** without altering the original source code.
- Example use case: Replacing a function for **unit testing**.

---

## 26. **`dir(p)` Function**

### What is `dir(p)`?
- `dir(p)` returns a list of **attributes and methods** of an object (functions, modules, strings, lists, dictionaries, etc.).

```python
# Example: dir() function
values = [10, 20, 30, 40]
print(dir(values))
```

---

## 27. **List Data Structure**

### Characteristics of Lists:
- Created using:
  - Square brackets `[]`
  - `list(p)` function
- Stores multiple values, including duplicates.
- Supports different data types.
- **Mutable** (modifiable).
- Follows **indexing**.
- **Preserves insertion order**.

---

## 28. **Common List Operations**
- **Adding an item**
- **Replacing an item**
- **Removing an item**

---

## 29. **`append(p)` Method**

### What is `append(p)`?
- A **predefined list method**.
- Adds an element **to the end** of the list.

```python
# Example: Using append()
values = [10, 20, 30, 40]
values.append(999)
print(values)
```

---

## 30. **`insert(p1, p2)` Method**

### What is `insert(p1, p2)`?
- A **predefined list method**.
- Inserts an element **at a specific index**.

```python
# Example: Using insert()
values = [10, 20, 30, 40]
values.insert(1, "Nireekshan")
print(values)
```

---

## 31. How to Reverse Values in a List?
### Using `reverse()` Method:
- `reverse()` is a predefined list method.
- It modifies the original list in place.

```python
values = [10, 20, 30, 40]
values.reverse()
print(values)  # Output: [40, 30, 20, 10]
```

### Using Slicing:
- `[::-1]` creates a reversed copy of the list.

```python
values = [10, 20, 30, 40]
reversed_values = values[::-1]
print(reversed_values)  # Output: [40, 30, 20, 10]
```

---

## 32. How to Check if a List is Empty?
- An empty list evaluates to `False` in conditional checks.

```python
values = []
if not values:
    print("List is empty")
else:
    print("List is not empty")
```

---

## 33. `isinstance(p1, p2)` Function
- Used to check the type of an object.
- Returns `True` if `p1` is an instance of `p2`.

```python
def check(obj):
    if isinstance(obj, str):
        print("This is a string")
    elif isinstance(obj, int):
        print("This is an integer")

check("Daniel")  # Output: This is a string
check(10)        # Output: This is an integer
```

---

## 34. List Comprehension
- A concise way to create lists from iterables.

```python
values = [1, 5, 9, 12, 13, 14]
result = [value+1 if value <= 10 else value+5 for value in values]
print(result)  # Output: [2, 6, 10, 17, 18, 19]
```

---

## 35. Differences Between List and Tuple
| Feature  | List (`[]`) | Tuple (`()`)|
|----------|------------|-------------|
| Mutability | Mutable (modifiable) | Immutable (unchangeable) |
| Size | Grows dynamically | Fixed size |

---

## 36. Set Data Structure
- Created using `{}` or `set()`.
- Stores **unique** values.
- Unordered and does not support indexing.

```python
my_set = {1, 2, 3, 4, 4, 4}
print(my_set)  # Output: {1, 2, 3, 4}
```

---

## 37. Differences Between List and Set
| Feature  | List (`[]`) | Set (`{}`) |
|----------|------------|------------|
| Order | Preserves order | No specific order |
| Duplicates | Allows duplicates | No duplicates |
| Indexing | Supports indexing | No indexing |

---

## 38. How to Remove Duplicates from a List?
```python
values = [10, 20, 30, 10, 10, 10]
unique_values = list(set(values))
print(unique_values)  # Output: [10, 20, 30]
```

---

## 39. Dictionary Data Structure
- Stores key-value pairs.
- Created using `{}` or `dict()`.

```python
details = {"name": "Daniel", "age": 25, "role": "Developer"}
print(details["name"])  # Output: Daniel
```

---

## 40. When to Use List, Tuple, Set, and Dictionary?
### **List (`[]`)**:
- Ordered collection
- Allows duplicates
- Supports indexing
- Mutable

### **Tuple (`()`)**:
- Ordered collection
- Allows duplicates
- Supports indexing
- **Immutable** (cannot be changed)

### **Set (`{}`)**:
- Unordered collection
- **No duplicate values**
- **No indexing**

### **Dictionary (`{key: value}`)**:
- Stores data as **key-value pairs**
- **No duplicate keys**

---

# OPPs

## 41. What is a Class in Python?
A **class** is a blueprint for creating objects and does not exist physically.

### Creating a Class:
```python
class Employee:
    def details(self):
        print("Hello, my name is Daniel")
```

### Key Points:
- A class can contain:
  - Constructor
  - Properties (variables)
  - Methods (functions inside a class)
- `self` is a predefined variable representing the current class object.

---
## 42. What is `self` in Python?
- `self` is a predefined variable in Python representing the current class object.
- It is used in:
  - Constructors
  - Instance variables
  - Instance methods

---
## 43. What is an Object?
An **object** is an instance of a class that allocates memory for class data members.

### Example:
```python
class Employee:
    def details(self):
        print("Hello, my name is Daniel")

emp = Employee()  # Creating an object
emp.details()      # Accessing method using object
```

### Key Notes:
- `emp` is an object of `Employee` class.
- Objects are used to access instance methods and variables.

---
## 44. What is a Constructor?
A **constructor** is a special method used to initialize instance variables when an object is created.

### Syntax:
```python
class Employee:
    def __init__(self):
        print("This is a constructor")

emp = Employee()
```

### Initializing Instance Variables:
```python
class Employee:
    def __init__(self, name, location):
        self.name = name
        self.location = location
    
    def details(self):
        print("My name is:", self.name)
        print("I am from:", self.location)

emp = Employee("Daniel", "India")
emp.details()
```

### Key Points:
- The constructor method is always named `__init__(self, ...)`.
- It is executed automatically when an object is created.

---
## 45. What are `try` and `except` in Python?
- **`try`**: Code that may raise an exception is written inside the `try` block.
- **`except`**: Handles exceptions if they occur inside the `try` block.

```python
try:
    num = int(input("Enter a number: "))
    result = 10 / num
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid input! Please enter a number.")
```

---
## 46. What is `with` in Python?
- The **`with`** statement ensures proper cleanup of resources like files and database connections.

```python
with open("sample.txt", "r") as file:
    data = file.read()
    print(data)
# No need to explicitly close the file
```

---
## 47. What is Pickling and Unpickling?
Pickling and unpickling involve saving and loading Python objects using the `pickle` module.

### Pickling (Saving an object to a file):
```python
import pickle

data = {"name": "Daniel", "age": 30}
with open("data.pkl", "wb") as file:
    pickle.dump(data, file)
```

### Unpickling (Loading an object from a file):
```python
with open("data.pkl", "rb") as file:
    loaded_data = pickle.load(file)
    print(loaded_data)
```

---
## 48. Steps to Work with Databases in Python
1. Import the database module.
2. Establish a connection.
3. Create a cursor object.
4. Execute SQL queries using built-in methods.
5. Commit or rollback transactions.
6. Fetch results from the cursor.
7. Close resources.

```python
import sqlite3

# Connect to database
conn = sqlite3.connect("example.db")
cursor = conn.cursor()

# Execute a query
cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
conn.commit()

# Close connection
conn.close()
```

---
## 49. What is Python Packaging (Egg/Wheel)?
- Python packaging helps in **distributing reusable code** across projects.
- `egg` and `wheel` are packaging formats used for distribution.
- Install a package using:
  ```bash
  pip install package_name
  ```

---
## 50. What are `import`, `from`, and `as` in Python?
### 1. `import`
Used to import modules.
```python
import math
print(math.sqrt(16))
```

### 2. `from`
Used to import specific functions/classes from a module.
```python
from math import sqrt
print(sqrt(16))
```

### 3. `as`
Used to assign an alias to a module or function.
```python
import numpy as np
arr = np.array([1, 2, 3])
print(arr)
```

