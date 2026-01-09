
# Introduction

---

## What is Python?

- **Python** is a general purpose and high-level programming language.
- **General purpose** means that many companies use Python to develop, test, and deploy software applications.
- **High-level programming** means it’s human-readable and easy to understand.

---

## Python Programming Applications

By using Python programming, we can develop:

1. **Standalone Applications:**  
   Applications that need to be installed on each machine to work.
2. **Web Applications:**  
   Applications that follow a client-server architecture.
3. **Database Applications:**  
   Applications that perform CRUD (create, update, retrieve, and delete) operations in a database.
4. **Big Data Applications:**  
   Applications that can process large datasets (e.g., using PySpark).
5. **Machine Learning:**  
   Applications that enable computers to learn automatically from past data.

---

## History of Python

- Python was created by Guido van Rossum in 1991.
- Python is open source software, meaning you can download it freely from [www.python.org](https://www.python.org) and customize the code as well.

**Python supports:**

- Both **functional** and **object-oriented** programming approaches.

---

# Keyword or Reserved Words (35)

A **keyword** (or reserved word) is a word that is reserved to perform a specific purpose in Python.

**Make a note:**

- All keywords in Python contain only alphabets.
- They are in lowercase except for three keywords: `True`, `False`, and `None`.

---

# Hello World Program

**Ways to write a Python program:**

You can write Python programs using many tools such as:
- Any text editor (Notepad, Notepad++, EditPlus)
- IDLE (Integrated Development Environment)
- Jupyter Notebook
- PyCharm IDE
- Anaconda
- Visual Studio
- Spyder
- Atom, etc.

**Python Program Execution Steps:**

- Open a text editor and write a Python program.
- Save the program with a `.py` or `.python` extension.
- Run (execute) the program.
- Finally, you will see the output.

```python
print("Hello World")
```

**Understanding the First Python Program:**

- `print()` is a predefined function in Python.
- The purpose of `print()` is to display the output.
- `"Hello World"` is a string in Python.

---

# Program Execution Flow

**Translator:**

A translator is a program that converts high-level code to low-level code.

**Examples:**

- **Compiler:** Converts the whole program in one step.
- **Interpreter:** Converts code line by line.

*Note:* Python is an interpreted programming language.

**Python Program Flow:**

```flow
Source Code => Byte Code => Python Virtual Machine => Output
```

- **Source Code:**  
  The written Python program, also called the source code.  
  Save the program with a `.py` or `.python` extension.

- **Running the Program:**  
  You run the program using the `py` command:
  ```bash
  py demo.py
  ```

- **Byte Code:**  
  While running the program, the interpreter converts the Python code into byte code (an intermediate file).  
  - This file is typically stored in the `__pycache__` folder in the current directory.
  - You can view the compiled byte code by running:
    ```bash
    python -m py_compile demo.py
    ```

- **Python Virtual Machine:**  
  The byte code is not directly understandable by the microprocessor.  
  The Python Virtual Machine converts the byte code into machine code that the processor can execute, producing the output.

---

# Naming Conventions

## 🧠 Identifiers

An **identifier** is a name used in a Python program. This name can refer to various entities such as:
- Package name
- Module name
- Variable name
- Function name
- Class name
- Method name

### Why Should We Follow Naming Conventions?
Following naming conventions in your code makes it:
- Easy to understand.
- Easy to read.
- Easy to debug.

### Points to Follow for Identifiers in Python:
1. **Allowed Characters:**  
   Only alphabets, numbers, and underscore (`_`) are allowed; otherwise, you'll encounter a `SyntaxError`.
2. **Starting Character:**  
   An identifier must not start with a number; otherwise, you'll get a `SyntaxError`.
3. **Case Sensitivity:**  
   Identifiers are case sensitive. Incorrect casing may result in a `NameError`.
4. **Keywords:**  
   You cannot use Python keywords as identifiers; doing so results in a `SyntaxError`.
5. **Spaces:**  
   Spaces are not allowed in identifiers; otherwise, you'll encounter a `SyntaxError`.

### Python Identifiers Table

1. **Class:**  
   - **Convention:** Use PascalCase (e.g., `MyClass`).  
   - **Note:** This rule applies to user-defined classes. Built-in class names are usually in lowercase.
2. **Package:**  
   - **Convention:** Names should be in lowercase.
3. **Module:**  
   - **Convention:** If the name has multiple words, use `snake_case` (e.g., `my_module`).
4. **Variable:**  
   - Use `snake_case` (e.g., `my_variable`).
5. **Function:**  
   - Use `snake_case` (e.g., `my_function`).
6. **Method:**  
   - Use `snake_case` (e.g., `my_method`).
7. **Non-public Instance Variables:**  
   - These should begin with an underscore (`_`) to indicate they are intended as private data.
8. **Constants:**  
   - **Convention:** Use uppercase letters.
   - If the constant name has multiple words, use `SNAKE_CASE` (e.g., `MY_CONSTANT`).
9. **Non-accessible Entities:**  
   - Some variables or class constructors use two underscores at the beginning and end to denote that they are private or non-accessible.

---

## Comments in a Program

1. **Single-line Comments:**  
   Use the `#` symbol.
2. **Multi-line Comments:**  
   Enclose the comment between triple quotes (`''' ... '''` or `""" ... """`).

---

# Variables

A **variable** is a reserved memory location used to store values. The purpose of a variable is to represent data (a collection of facts such as alphabets, numbers, alphanumeric characters, and symbols).

### Properties of Variables:
- **Name**
- **Value**
- **Type**

### Variable Re-initialization:
- Re-initializing a variable replaces its old value with a new one.

---

# Data Type

A **data type** represents the type of data stored in a variable or memory.

### Checking Data Types:
- Use the `type()` function to check the type of a variable.

### Built-in Data Types in Python:
- **int:** Represents numbers without decimal values.
- **float:** Represents numbers with decimal values.
- **bool:** Represents Boolean values in Python (`True` or `False`).
- **None:** Represents an object that does not contain any value.
- **Sequences:**  
  - `str` (string)
  - `list`
  - `tuple`
  - `set`
  - `dict`
  - `range`

**Additional Details:**
- **String:**  
  A string refers to a group of characters enclosed within quotes.
- **Range Data Type:**  
  - `range(p)` or `range(start, end)`

---

# Operators

An **operator** is a symbol that performs an operation. It acts on some variables, which we call *operands*. Operators can be classified into:

- **Unary Operators**
- **Binary Operators**
- **Ternary Operators**

> **Note:** Python does not have increment (`++`) and decrement (`--`) operators.

---

## Arithmetic Operators

Arithmetic operators perform basic mathematical operations. Some common ones include:

- `+` : Addition  
- `-` : Subtraction  
- `*` : Multiplication  
- `/` : Division  
  - **Note:** The division operator (`/`) always performs floating point arithmetic, so it returns float values.
- `%` : Modulus (remainder of division)
- `**` : Exponentiation (exponential power)
- `//` : Floor Division  
  - **Note:** Floor division returns only the integer quotient if both operands are integers; otherwise, it returns a float.

---

## Assignment Operators

Assignment operators are used to assign values to variables. Examples include:

- `=`  
- `+=`  
- `-=`  
- `*=`  
- `/=`  
- `%=`  
- `**=`  
- `//=`

---

## Unary Minus Operator

- The unary minus operator (`-`) is used to indicate a negative value.  
- **Example:**  
  ```python
  x = -5  # assigns negative five to x

---


# Input and Output

- **Input** represents data given to the program.  
- **Output** represents the result of the program.  

---

## Hard Coding

- Till now, we have executed examples by hard coding values into variables.  
- In this chapter, we will learn how to take values at runtime.

### Hard coding the values

```python
age = 16
```

- Based on the requirement, we can take values at runtime or dynamically as well.

```
Please enter the age: 16
Entered value is: 16
```

---

## `input(p)` Function

- `input(p)` is a predefined function.
- This function accepts input from the keyboard.
- It takes a value from the keyboard and returns it as a **string** type.
- Based on the requirement, we can convert from **string** to other types.

### Example: Printing Name by Taking Value at Runtime

```python
name = input("Enter the name: ")
print("You entered name as:", name)
```

### Example: Checking Return Type of `input()` Function

```python
value = input("Enter the value: ")
print("Entered value as:", value)
print("Type is:", type(value))
```

---

## Convert from String to Integer

- We can convert a **string** value into an **integer** value.
- `int(p)` is a predefined function.
- This function converts a value into the **int** data type.

### Example: String to Integer Conversion

```python
a = "123"
b = int(a)
print(type(b))  # Output: <class 'int'>
```

---

## Convert from String to Float

- We can convert a **string** value into a **float** value.
- `float(p)` is a predefined function.
- This function converts a value into the **float** data type.

### Example: String to Float Conversion

```python
a = "123.99"
b = float(a)
print(type(b))  # Output: <class 'float'>
```

---

## Convert Float Data Type to Integer

- `int(p)` is a predefined function in Python.
- This function converts a **float** value into an **int** value.

### Example: Float to Integer Conversion

```python
a = 10000.45
b = int(a)
print(b)  # Output: 10000
```

---

## Type Conversion Functions in Python

Below is a list of commonly used type conversion functions:

| Function  | Description |
|-----------|-------------|
| `int(p)`  | Converts other data types into an integer. |
| `float(p)` | Converts other data types into a float. |
| `str(p)`  | Converts other data types into a string. |
| `bool(p)` | Converts other data types into a boolean. |
| `list(p)` | Converts a sequence into a list. |
| `tuple(p)` | Converts a sequence into a tuple. |
| `set(p)`  | Converts a sequence into a set. |
| `dict(p)` | Converts a tuple of order `(key, value)` into a dictionary. |

---


# Flow Control

- The order of statements execution is called **flow of control**.

## Types of Execution

- The program's statements can execute in several ways based on requirements. They can run sequentially, conditionally, or repeatedly.

- In any programming language, statements will be executed mainly in three ways:
  - **Sequential execution**
  - **Conditional execution**
  - **Looping execution**

---

## 1. Sequential Execution

- Statements execute from **top to bottom**, meaning **one by one sequentially**.
- Sequential statements are useful for simple programs.

### Example: Sequential Execution

```python
print("one")
print("two")
print("three")
```

---

## 2. Conditional Execution

- Based on **conditions**, statements will be executed.
- Conditional statements help develop **better and more complex programs**.

### Conditional (Decision-Making) Statements:

| Statement Type | Valid Combinations |
|---------------|-------------------|
| `if` | ✅ Valid |
| `if else` | ✅ Valid |
| `if elif else` | ✅ Valid |
| `if elif elif else` | ✅ Valid |

---

## 3. Looping Execution

- **Looping** executes statements **randomly and repeatedly based on conditions**.
- It is useful for developing complex programs.

### Types of Loops:

- `for` loop
- `while` loop

### Other Keywords:

- `break`
- `continue`
- `pass`

---

## Indentation in Python

- **Python uses indentation to indicate a block of code**.
- Indentation means **adding white spaces** before a statement.
- Python uses **4 spaces** as indentation **by default**.
- If indentation is incorrect, Python throws an **IndentationError**.

```python
IndentationError: expected an indented block
```

---

# Conditional or Decision-Making Statements

## 1. `if` Statement

```python
if condition:
    # if block statements
out_of_if_block_statements
```

- `if` is a **keyword** in Python.
- The `if` statement contains an **expression/condition/value**.
- **Colon (`:`) is mandatory**, otherwise, it throws **SyntaxError**.
- If the **condition is True**, the `if` block executes.
- If the **condition is False**, the `if` block is skipped.

### Example: `if` Statement

```python
x, y = 1, 1
print("x == y value is:", (x == y))

if x == y:
    print("if block statements executed")
```

---

## 2. `if else` Statement

```python
if condition:
    # if block statements
else:
    # else block statements
```

### Example: `if else` Statement

```python
x = 1
y = 1
print("x == y value is:", (x == y))

if x == y:
    print("if block statements executed")
else:
    print("else block statements executed")
```

---

## 3. `if elif else` Statement

```python
if condition1:
    # if block statements
elif condition2:
    # elif block statements
elif condition3:
    # elif block statements
else:
    # else block statements
```

- **`if`, `elif`, and `else` are keywords in Python**.
- **Colon (`:`) is mandatory**, otherwise, **SyntaxError** occurs.
- The first **True** condition executes, skipping the rest.
- If **all conditions are False**, the `else` block executes.

### Example: `if elif else` Statement

```python
print("Please enter values from 0 to 4")
x = int(input("Enter a number: "))

if x == 0: 
    print("You entered:", x)
elif x == 1: 
    print("You entered:", x)
elif x == 2:
    print("You entered:", x)
elif x == 3:
    print("You entered:", x)
elif x == 4: 
    print("You entered:", x)
else:
    print("Beyond the range specified")
```

---

# Looping

- If we want to **execute a group of statements multiple times**, we use **looping**.
- Two types:
  - **`for` loop**
  - **`while` loop**

---

## 1. `for` Loop

- The `for` loop **iterates** over a sequence (string, list, tuple, etc.).
- Syntax:

```python
for variable in sequence:
    statements
```

### Example: Using `for` Loop

```python
values = [10, 20, 30, "Daniel"]
for value in values:
    print(value)
```

---

## 2. `while` Loop

- The `while` loop **executes a block of code while a condition is True**.
- Syntax:

```python
initialization
while condition:
    statements
    increment/decrement
```

### Example: Using `while` Loop

```python
x = 1
while x <= 5:
    print(x)
    x = x + 1
```

---

# `break` Statement

- `break` is used **inside loops** to **terminate execution early**.

### Example: `while` Loop without `break`

```python
x = 1
while x <= 10:
    print("x =", x)
    x = x + 1
print("Out of loop")
```

### Example: Using `break` in `while` Loop

```python
x = 1
while x <= 10:
    print("x =", x)
    x = x + 1
    if x == 5:
        break
print("Out of the loop")
```

---

# `continue` Statement

- `continue` is used **to skip the current iteration** and proceed to the next one.

### Example: Using `continue` in a Loop

```python
cart = [10, 20, 500, 700, 50, 60]
for item in cart:
    if item == 500:
        continue  # Skips 500
    print(item)
```

---

# `pass` Statement

- `pass` is used **as a placeholder for future code**.
- It **does nothing** when executed.

### Example: Function Without `pass`

```python
def upcoming_sales():
upcoming_sales()

# Output:
IndentationError: expected an indented block
```

### Example: Function With `pass`

```python
def upcoming_sales():
    pass
upcoming_sales()
```

- `pass` can also be used inside **empty classes, loops, and methods**.

---

# STRING

## What is a string?

### 1. Definition 1
- A group of characters enclosed within quotes (single, double, or triple) is called a string.

### 2. Definition 2
- A string is a sequential collection of characters.

### 3. String is more popular
- In any kind of programming language, the most commonly used data type is a string.

## Creating string

```python
# With single quotes
name1 = 'Daniel'

# With double quotes
name2 = "Daniel"

# With triple single quotes
name3 = '''Daniel'''

# With triple double quotes
name4 = """Daniel"""
```

### Make a note
- Generally, double quotes are most commonly used to create a string.

### 1. When should we use triple single and triple double quotes?
- If you want to create multiple lines of string, then triple single or triple double quotes are the best to use.

```python
# Printing Employee information
loc1 = '''TCS company
	White Field
	Bangalore'''
```

## Indexing

### Accessing string by using index

```python
wish = "Hello World"
print(wish[0])
print(wish[1])
```

### Accessing string by using slicing

```python
wish = "Hello World"
print(wish[0:7])
```

### Accessing string by using for loop

```python
wish = "Hello World"
for char in wish:
	print(char)
```

## Mutable vs Immutable

### Mutable
- Once we create an object, its state can be changed, modified, or updated.
- This behavior is called mutability.

### Immutable
- Once we create an object, its state cannot be changed, modified, or updated.
- This behavior is called immutability.

### Strings are immutable
- Strings have an immutable nature.
- Once we create a string object, we cannot change or modify the existing object.

```python
# Printing name and first index in string
name = "Daniel"
print(name)
print(name[0])

# String having immutable nature
name = "Daniel"
name[0] = "X"  # This will raise TypeError
```

```
TypeError: 'str' object does not support item assignment
```

## Mathematical operators on string objects

- We can perform two mathematical operations on strings:
  - Addition (`+`) operator.
  - Multiplication (`*`) operator.

### 1. Addition (`+`) operator with string
- The `+` operator works like concatenation or joins the strings.

```python
# + works as concatenation operator
a = "Python"
b = "Programming"
print(a + b)
```

### 2. Multiplication (`*`) operator with string
- This operator works with strings to perform repetition.

```python
# * operator works as repetition in strings
course = "Python"
print(course * 3)
```

## Length of the string
- We can find the number of characters in a string by using the `len()` function.

```python
# Length of the string
course = "Python"
print(len(course))
```

## Membership operators (`in`, `not in`)

### Definition 1
- We can check if a string or character is a member of a string using `in` and `not in` operators.

### Definition 2
- We can check if a string is a substring of the main string using `in` and `not in` operators.

### 1. `in` operator
- Returns `True` if the string or character is found in the main string.

```python
# in operator
print('p' in 'python')
print('pa' in 'python')
```

### 2. `not in` operator 
- Returns the opposite result of the `in` operator.
- Returns `True` if the string or character is not found in the main string.

```python
# not in operator
print('b' not in 'apple')
```

## Methods in `str` class

- `str` is a predefined class.
- The `str` class contains methods.
- We can check these methods using the `dir()` function.
- `str` class contains two types of methods:
  - Methods with underscore symbols (We do not need to focus on these).
  - Methods without underscore symbols (We need to focus on these).

```python
# Printing str class methods by using dir(str) function
print(dir(str))
```

### Important point
- As per object-oriented principles:
  - To access instance methods, we should use the object name.
- All `str` class methods can be accessed using `str` objects.

## Important methods in `str` class

### `upper()`
- Converts lowercase letters into uppercase.

```python
# Converting from lowercase to uppercase
name = "daniel"
print("After converting: ", name.upper())
```

### `lower()`
- Converts uppercase letters into lowercase.

```python
# Converting from uppercase to lowercase
name = "DANIEL"
print("After converting: ", name.lower())
```

### `strip()`
- Removes left and right spaces of a string but does not remove spaces in the middle.

```python
# Removing spaces at the start and end of the string
course = "Python            "
x = course.strip()
print("After removing spaces, course length is: ", len(x))
```

### `count(p)`
- Finds the number of occurrences of a substring in a string.

```python
# Counting substring occurrences using count() method
s = "Python programming language, Python is easy"
print(s.count("Python")) 
print(s.count("Hello")) 
```

### `replace(p1, p2)`
- Replaces an old string with a new string.
- Creates a new string object instead of modifying the existing one.

```python
# Replacing string using replace() method
s1 = "Java programming language"
s2 = s1.replace("Java", "Python")
```

### `split(p)`
- Splits a string based on the specified separator and returns a list.

```python
# Splitting a string using split() method
s = "Python programming language" 
n = s.split()

# Splitting string using split(p) method
s = "This is, Python programming, language " 
n = s.split(",")
```

---

# FUNCTIONS

## Function

- A function can contain a group of statements that perform a task.

### Advantages

- Maintaining the code is easy.
- Code reusability.

### Make a Note

- `print()` is a predefined function in Python that prints output on the console.

## Types of Functions

There are two types of functions:

1. **Pre-defined or built-in functions**
2. **User-defined functions**

### 1. Pre-defined or Built-in Functions

- The functions that already exist in Python are called predefined functions.

**Examples:**

```python
print(p)
type(p)
input(p)
```

### 2. User Defined Functions

- Based on the requirement, a programmer can create their own function. These functions are called user-defined functions.

## Function Related Terminology

To understand the function concept better, we need to focus on function-related terminology:

- `def` keyword
- Name of the function
- Parenthesis `()`
- Parameters (if required)
- Colon `:` symbol
- Function body
- Return type (optional)

## Function Definition

- A function can contain a group of statements.
- The purpose of a function is to perform an operation.
- A function has two main parts:
  1. **Creating a function**
  2. **Calling a function**

### 1. Creating a Function

- The first step is to create a function using the `def` keyword.
- After `def`, write the function name.
  - After the function name, write parentheses `()`.
  - This parenthesis may contain parameters.
    - Parameters are like variables that receive values.
    - If a function has parameters, we need to provide values when calling it.
  - After parentheses, write a colon `:`.
  - Provide indentation for the function body.
- The function body contains the logic.
- Before closing the function, it may contain a return type.

**Syntax:**

```python
def function_name():
    """ Docstring """  
    # Body of the function to perform operations
```

### Naming Convention for Functions

- Function names should be in lowercase.
- If the name has multiple words, separate them using an underscore (`_`).

**Example:**

```python
def display():
    print("Welcome to function")
```

### Make a Note

- When we execute the above program, the function body is not executed.
- To execute the function body, we need to **call the function**.

## 2. Calling a Function

- After creating a function, we need to call it to execute its body.
- The function name should match exactly when calling, otherwise, we will get an error.

**Example:**

```python
def display():
    print("Welcome to function concept")

display()  # Function call
```

```python
detail()  # This will cause an error as 'detail' is not defined
```

### Can I Create More Than One Function in a Single Python Program?

- Yes, we can create multiple functions based on the requirement.

**Example:**

```python
def first():
    print("This is the first function")

def second():
    print("This is the second function")

first()
second()
```

## A Function Can Call Another Function

- A function can call another function as well.

**Example:**

```python
def m1():
    print("First function")
    m2()

def m2():
    print("Second function")

m1()
```

## Functions Based on Parameters

Functions can be categorized into two types based on parameters:

1. **Function without parameters**
2. **Function with parameters**

### 1. Function Without Parameters

- A function with no parameters is called a **No parameterized function**.

**Example:**

```python
def display():
    print("Welcome to a function with no parameters")

display()
```

### 2. Function With Parameters

- Function parameters help process operations.
- When we pass parameters:
  - The function captures parameter values.
  - These values perform operations.
  - Finally, the function brings the result.

**Example:**

```python
def testing(a):
    print("One parameterized function:", a)

testing(10)
testing(10.56)
testing("Daniel")
```

**Example of Addition Function:**

```python
def addition(a, b):
    print("Addition of two values:", a + b)

addition(10, 20)
```

## `return` Keyword in Python

Based on the return statement, functions can be divided into two types:

1. **Function without return statement**
2. **Function with return statement**

- `return` is a keyword in Python.
- A function with a `return` statement is valid.

### Function Without Return Statement

```python
def balance():
    print("My bank balance is:")

balance()
```

### Function With Return Statement

```python
def balance():
    print("My bank balance is:")
    return 100

b = balance()
print("Balance is:", b)
```

### Why Assign a Function Call to a Variable?

- If we assign a function call to a variable, that variable holds the returned value, which we can use further in our program.

```python
def balance():
    return 100

b = balance()

if b == 0:
    print("Balance is:", b)
elif b < 0:
    print("Balance is:", b, "negative, please deposit")
else:
    print("Balance is:", b)
```

## Lambda Functions (Anonymous Functions)

- `lambda` is a keyword in Python to create **anonymous functions**.
- An **anonymous function** is a function without a name.
- Lambda functions process input and return a result.

**Syntax:**

```python
lambda arguments: expression
```

**Example:**

```python
s = lambda a: a * a
x = s(4)
print(x)  # Output: 16
```

### Lambda Function vs Normal Function

**Normal Function:**

```python
def square(t):
    return t * t

s = square(2)
print(s)
```

**Lambda Function:**

```python
add = lambda x, y: x + y
result = add(1, 2)
print(result)
```

## `map()`, `filter()`, and `reduce()` Functions with Lambda

### 1. `map()` Function

```python
without_gst_cost = [100, 200, 300, 400]
with_gst_cost = list(map(lambda x: x + 10, without_gst_cost))
print(with_gst_cost)
```

### 2. `filter()` Function

```python
items_cost = [999, 888, 1100, 1200, 1300, 777]
gt_thousand = list(filter(lambda x: x > 1000, items_cost))
print(gt_thousand)
```

### 3. `reduce()` Function

```python
from functools import reduce

items_cost = [111, 222, 333, 444]
total_cost = reduce(lambda x, y: x + y, items_cost)
print(total_cost)
```

---

# Modules in Python

## What is a Module?
- In Python, a module is a saved Python file.
- This file can contain a group of classes, methods, functions, and variables.
- Every Python file with a `.py` or `.python` extension is called a module.

### Example: `additionmultiplication.py`
```python
x = 10

def addition(a, b):
    print("Sum of two values:", (a + b))

def multiplication(a, b):
    print("Multiplication of two values:", (a * b))
```
- Now, `additionmultiplication.py` is a module containing one variable and two functions.

## Importing Modules
- `import` is a keyword in Python.
- Using `import`, we can include modules in our program.
- Once imported, we can use the module's members (variables, functions, etc.).

### Example: Importing `additionmultiplication` module
```python
import additionmultiplication

print(additionmultiplication.x)
additionmultiplication.addition(1, 2)
additionmultiplication.multiplication(2, 3)
```
**Note:** When a module is used in a program, a compiled file is generated and stored permanently on the hard disk.

## Renaming (Aliasing) a Module
- `as` is a keyword used to rename a module.

### Syntax:
```python
import additionmultiplication as amul
```
- Now, we can access the module members using `amul` instead of `additionmultiplication`.

### Example:
```python
import additionmultiplication as amul

print(amul.x)
amul.addition(1, 2)
amul.multiplication(3, 4)
```

## Using `from` and `import`
- `from` allows us to import specific members of a module.
- The advantage is that we can use the members directly without using the module name.

### Example:
```python
from additionmultiplication import x, addition

print(x)
addition(10, 20)
```

If we try to access a member that was not imported:
```python
from additionmultiplication import x, addition

print(x)
multiplication(10, 20)  # NameError: name 'multiplication' is not defined
```

## Importing All Members Using `*`
- We can use `*` to import all members of a module.

### Example:
```python
from additionmultiplication import *

print(x)
addition(10, 20)
multiplication(10, 20)
```

## Aliasing Module Members
- We can give alias names to module members.

### Example:
```python
from additionmultiplication import x as y, addition as add

print(y)
add(10, 20)
```

---

# Python Packages

## What is a Package?
- A package is a folder or directory containing a collection of Python modules.

### `__init__.py` File
- Any folder containing `__init__.py` is considered a Python package.
- `__init__.py` can be an empty file.
- A package can also contain sub-packages.

### Advantages of Packages
- Resolves naming conflicts.
- Identifies components uniquely.
- Improves modularity of applications.

## Example 1: Simple Package Structure
```
|
|--- demo1.py
|
|--- demo2.py
|
|--- __init__.py
|
|--- pack1
    |
    |--- test1.py
    |
    |--- __init__.py
```

### `test1.py`
```python
def m1():
    print("Hello, this is test1 present in pack1")
```

### `demo1.py`
```python
import pack1.test1
pack1.test1.m1()
```

### `demo2.py`
```python
from pack1.test1 import m1
m1()
```

## Example 2: Package with Subdirectory
```
|
|--- demo3.py
|
|--- __init__.py
|
|--- maindir
    |
    |--- test2.py
    |
    |--- __init__.py
    |
    |--- subdir
        |
        |--- test3.py
        |
        |--- __init__.py
```

### `test2.py`
```python
def m2():
    print("This is test2 present in maindir")
```

### `test3.py`
```python
def m3():
    print("This is test3 present in maindir.subdir")
```

### `demo3.py`
```python
from maindir.test2 import m2
from maindir.subdir.test3 import m3

m2()
m3()
```

## Summary: Libraries, Packages, and Modules
- **Library** → A group of packages.
- **Package** → A group of modules.
- **Module** → A group of variables, functions, and classes.

---


# List Data Structure

## Why should we learn about data structures?

- The common requirement in any real-time project is like creating, updating, retrieving, and deleting elements.
- Few more real-time operations are:
  - Storing
  - Searching
  - Retrieving
  - Deleting
  - Processing
  - Duplicate
  - Ordered
  - Unordered
  - Size
  - Capacity
  - Sorting
  - Un-sorting
  - Random access
  - Keys
  - Values
  - Key-value pairs

So, to understand the above operations—where to use and how to use them—we need to learn about data structures.

## Python Data Structures

- If you wanted to store a group of individual objects in a single entity, then you should go for data structures.

### Sequence of Elements

- A data structure is also called a sequence.
- A sequence is a datatype that can contain a group of elements.
- The purpose of any sequence is to store and process a group of elements.
- In Python, strings, lists, tuples, sets, and dictionaries are very important sequence datatypes.

### List Data Structure

- We can create a list using:
  - Square brackets `[]`
  - `list()` predefined function.
- A list can store a group of objects or elements.
  - A list can store the same (Homogeneous) type of elements.
  - A list can store different types (Heterogeneous) of elements.
- A list size will increase dynamically.
- In a list, the insertion order is preserved.
  - Example: 
    - Input: `[10, 20, 30]`
    - Output: `[10, 20, 30]`
- Duplicate elements are allowed.
- Lists have mutable nature.
  - Mutable means once we create a list object, we can change or modify the content of the list object.
- Store elements by using indexes.
  - A list supports both positive and negative indexes:
    - Positive index: from left to right
    - Negative index: from right to left

#### Note:
- A list is a predefined class in Python.
- Once we create a list object, an internal object is created for the list class.

### Creating Lists

1. **Creating an Empty List**
   ```python
   # Creating empty list
   a = []
   ```

2. **Creating a List with Elements**
   ```python
   # Creating list with different types of elements
   student_info = ["Daniel", 10, 35.9]
   print(student_info)
   ```

3. **Creating List Using `list(p)` Function**
   - `list(p)` is a predefined function in Python.
   - This function takes only one parameter.
   - The parameter must be a sequence (e.g., range, list, set, tuple, etc.) object.
   
   ```python
   # Creating list using list(p) function
   r = range(0, 10)
   a = list(r)
   print(a)
   ```

### List Having Mutable Nature

- Once we create a list object, we can modify the elements in the existing list.
  
```python
# List having mutable nature
a = [1, 2, 3, 4, 5]
print(a)
a[0] = 20
print(a)
```

### Accessing Elements from List

1. **By Using Index**
   - Index represents accessing the elements by their position in the list.
   - Index starts from 0 onwards.
   - List supports both positive and negative indexes.
     - Positive index: from left to right
     - Negative index: from right to left
   - If we try to access beyond the list's range, we will get an `IndexError`.

   ```python
   # List indexing
   names = ["Daniel", "Prasad", "Ramesh"]
   print(names[0])
   print(names[1])
   
   # IndexError: list index out of range
   names = ["Daniel", "Prasad", "Ramesh"]
   print(names)
   print(names[30])
   ```

2. **Slicing**
   - Slicing represents extracting a piece of the list from an already created list.
   
   Syntax: `[start: stop: stepsize]`
   - `start`: Index where slice starts (default is 0)
   - `stop`: Index where slice ends (default is max index of list)
   - `stepsize`: Increment value (default is 1)
   
   ```python
   # Slice example
   n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   print(n)
   print(n[:])
   print(n[::])
   print(n[0:5:])
   ```

3. **Accessing List by Using For Loop**

```python
# Accessing elements from list by using for loop
values = [100, 200, 300, 400]

for value in values:
    print(value)
```

### `len(p)` Function

- Use the `len(p)` function to find the length of a list. It returns the number of elements in the list.

```python
# To find the length of list
values = [10, 20, 30, 40, 50]
print(len(values))
```

### Methods in List Data Structure

- Lists are a predefined class, so they can contain methods.

#### Important Methods in List

- `count(p)`: Returns the number of occurrences of a specific value in the list.
- `append(p)`: Adds an object to the end of the list.
- `insert(p1, p2)`: Inserts a value at a specific position in the list.
- `remove()`: Removes a value from the list.

Ordering Elements of List:

- `reverse()`: Reverses the values in the list.
- `sort()`: Sorts the values in the list (ascending for numbers, alphabetical for strings).

#### Example Usage:

```python
# To find the count of a specific value in a list
n = [1, 2, 3, 4, 5, 5, 5, 3]
print(n.count(5))
print(n.count(2))

# Appending elements into the list
a = []
a.append(10)
a.append(20)
a.append(30)

# Inserting elements into the list
a = [10, 20, 30, 40, 50]
a.insert(0, 76)

# Removing element from the list
a = [10, 20, 30]
a.remove(10)

# Reversing the list
a = [10, 20, 30, 40]
print(a)
a.reverse()

# Sorting the numbers and names
a = [10, 40, 50, 20, 30]
a.sort()

b = ['Daniel', 'Ramesh', 'Arjun']
b.sort()
```

### Mathematical Operators: `+` and `*`

1. **Concatenation Operator (`+`)**

   - The `+` operator concatenates two list objects to join them and returns a single list.

   ```python
   # + operator concatenates the lists
   a = [10, 20, 30]
   b = [40, 50, 60]
   c = a + b
   ```

2. **Repetition Operator (`*`)**

   - The `*` operator repeats the elements in the list.

   ```python
   # * operator repeats the lists
   a = [10, 20, 30]
   print(a)
   print(a*2)
   ```

### Membership Operators

- We can check if an element is a member of a list using the `in` and `not in` operators.

```python
# Membership operators
a = [10, 20, 30, 40, 50]

print(20 in a)  # True
print(20 not in a)  # False

print(90 in a)  # False
print(90 not in a)  # True
```

### List Comprehension

- List comprehension represents creating new lists from an iterable object like a list, set, tuple, dictionary, and range.
- It is a concise way of applying conditional logic on every item in the iterable to return a new list.

```python
# List comprehension examples
values = [10, 20, 30]
result = [value + 2 for value in values]

values = [10, 20, 30]
result = [value * 3 for value in values]

values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
result = [value for value in values if value <= 50]

# Square numbers from 1 to 10 using list comprehension
values = range(1, 11)
squares = [value * 2 for value in values]
```

---


# Tuple Data Structure in Python

## Introduction

A tuple is a collection data structure in Python that allows you to store multiple items in a single variable. It is similar to a list but with the key difference that tuples are **immutable**.

### Creating a Tuple

You can create a tuple using:
- Parentheses `()` symbols
- The predefined `tuple(p)` function

### Key Properties of Tuples

1. **Order Preservation:** Tuples maintain the order of elements.
   - Example: If you insert `(10, 20, 30)`, the output will be `(10, 20, 30)`.

2. **Duplicate Elements:** Tuples allow duplicate elements.
   - Example: `(10, 20, 20)` is a valid tuple.

3. **Immutable:** Once created, a tuple cannot be modified (i.e., it is immutable).  
   - Example: You cannot change elements in a tuple after creation.

4. **Indexed:** Elements in a tuple are stored by index, both positive and negative indices are supported.
   - Positive index: From left to right (e.g., `t[0]`)
   - Negative index: From right to left (e.g., `t[-1]`)

---

## When to Use Tuples?

- **Immutable Data:** Use tuples when the data should not change, such as:
   - Weekdays
   - Month names
   - Constant configuration values

---

## Types of Tuples

### 1. Tuples with Same Type of Objects

```python
employee_ids = (10, 20, 30, 40, 50)
print(employee_ids)
```

### 2. Tuples with Different Types of Objects

```python
employee_info = (10, "Daniel", 35.5)
print(employee_info)
```

### 3. Single-Value Tuples

A single-value tuple requires a trailing comma to be recognized as a tuple.

```python
# Incorrect (not a tuple)
number = (9)
print(type(number))  # <class 'int'>

# Correct (tuple)
single_value = (9,)
print(type(single_value))  # <class 'tuple'>
```

---

## Parentheses are Optional

Parentheses are not always required to create a tuple:

```python
emp_ids = 10, 20, 30, 40
print(emp_ids)  # (10, 20, 30, 40)
```

---

## Ways to Create Tuples

### 1. Empty Tuple

You can create an empty tuple using empty parentheses:

```python
emp_id = ()
```

### 2. Tuple with Multiple Values

A tuple can hold a mix of elements, including integers, strings, and more:

```python
emp_id = (11, 12, 13)
std_id = 120, 130, 140
t = (11, 12, 13, "Daniel")
```

### 3. Using the `tuple()` Function

You can also create a tuple by converting other data structures like lists:

```python
a = [11, 22, 33]
t = tuple(a)
```

---

## Accessing Tuple Elements

You can access elements in a tuple using:

- **Indexing:** Access elements by their position.
- **Slicing:** Extract a range of elements from the tuple.

### 1. Indexing

```python
t = (10, 20, 30, 40, 50, 60)
print(t[0])  # 10
print(t[-1])  # 60
```

### 2. Slicing

Slicing allows you to retrieve a part of the tuple.

```python
t = (10, 20, 30, 40, 50, 60)
print(t[2:5])  # (30, 40, 50)
print(t[::2])  # (10, 30, 50)
```

---

## Tuple Immutability

Tuples are immutable, meaning you cannot change their contents after creation.

```python
t = (10, 20, 30, 40)
t[1] = 70  # TypeError: 'tuple' object does not support item assignment
```

---

## Mathematical Operators on Tuples

You can perform mathematical operations like concatenation (`+`) and repetition (`*`) on tuples.

### 1. Concatenation (`+`)

The `+` operator combines two tuples into one.

```python
t1 = (10, 20, 30)
t2 = (40, 50, 60)
t3 = t1 + t2
print(t3)  # (10, 20, 30, 40, 50, 60)
```

### 2. Repetition (`*`)

The `*` operator repeats the tuple elements a specified number of times.

```python
t1 = (10, 20, 30)
t2 = t1 * 3
print(t2)  # (10, 20, 30, 10, 20, 30, 10, 20, 30)
```

### 3. Length of Tuple (`len()`)

You can find the number of elements in a tuple using the `len()` function:

```python
t = (10, 20, 30, 40)
print(len(t))  # 4
```

---

## Tuple Methods

Tuples come with two primary methods:

### 1. `count(p)`

Returns the number of occurrences of a specified item.

```python
t = (10, 20, 10, 10, 20)
print(t.count(10))  # 3
```

### 2. `index(p)`

Returns the index of the first occurrence of the specified item. If the item is not found, it raises a `ValueError`.

```python
t = (10, 20, 30)
print(t.index(30))  # 2
print(t.index(77))  # ValueError: tuple.index(x): x not in tuple
```

You can explore all methods associated with a tuple using the `dir()` function:

```python
print(dir(tuple))  # List all tuple methods
```

---

## Differences Between Lists and Tuples

| Feature                  | List                         | Tuple                        |
|--------------------------|------------------------------|------------------------------|
| **Syntax**               | `[]`                         | `()` (optional)              |
| **Mutability**           | Mutable                      | Immutable                    |
| **Element Modification** | Can modify elements          | Cannot modify elements       |
| **Use Case**             | Use when data changes        | Use when data doesn't change |

---

## Can We Add Elements to a Tuple?

While you cannot modify the tuple itself, you can modify the list elements within a tuple (if the tuple contains lists).

```python
t = (11, 22, [33, 44], 55, 66)
t[2].append(77)  # This works because the list is mutable
print(t)  # (11, 22, [33, 44, 77], 55, 66)
```

---

## Conclusion

Tuples are a powerful data structure when you need to store a fixed collection of items that should not change over time. They are fast, efficient, and useful in scenarios where immutability is important, such as storing constant values or as dictionary keys.

---


# Set Data Structure in Python

## Introduction

A **set** is an unordered collection data structure in Python that holds a unique set of elements. It can store both homogeneous (same type) and heterogeneous (different types) elements but does not preserve the order of insertion.

### Key Properties of Sets

1. **Unordered:** The order of elements in a set is not guaranteed.
   - Example: Input `{10, 20, 30, 40}` could output `{20, 40, 10, 30}` (order is not preserved).

2. **No Duplicates:** Sets automatically remove duplicate elements.
   - Example: `{10, 20, 20, 30}` becomes `{10, 20, 30}`.

3. **Mutable:** Sets are mutable, meaning you can modify their elements after creation.

4. **No Indexing:** Sets do not support indexing, slicing, or accessing elements by position.

---

## When to Use a Set?

- Use sets when you need to store a collection of unique values.
- Sets do not allow duplicate elements and do not preserve insertion order.
- Useful when checking for membership or removing duplicates from a list.

---

## Creating a Set

### 1. Using Curly Braces `{}`

You can create a set by using curly braces with elements separated by commas:

```python
# Creating a set of integers
s = {10, 20, 30, 40}
print(type(s))  # <class 'set'>

# Creating a set with different types of elements
s = {10, "Daniel", 30.9, "Prasad", 40}
print(type(s))  # <class 'set'>
```

### 2. Using the `set()` Function

You can also create a set using the `set()` function, which accepts a sequence (list, tuple, or other iterable) as its argument:

```python
# Creating a set from a range
r = range(0, 10)
s = set(r)
print(s)
```

### 3. Empty Set

To create an empty set, you must use `set()` — not `{}`, as `{}` creates an empty dictionary.

```python
# Creating an empty set
empty_set = set()
print(empty_set)  # set()
```

---

## Set Characteristics

- **Dynamic Size:** Sets grow dynamically as elements are added.
- **No Indexing:** You cannot access elements by index or position.
  
### Example: Adding and Removing Elements

```python
s = {10, 20, 30}
s.add(40)  # Adding an element
print(s)  # {40, 10, 20, 30}

s.remove(20)  # Removing an element
print(s)  # {40, 10, 30}
```

If you attempt to remove an element that doesn’t exist, it will raise a `KeyError`.

```python
s.remove(100)  # KeyError: 100 not in set
```

---

## Set Methods

Sets come with several built-in methods to modify or query their contents:

### 1. `add(p)`

Adds an element to the set.

```python
s = {10, 20, 30}
s.add(40)  # {10, 20, 30, 40}
```

### 2. `remove(p)`

Removes the specified element from the set. Raises a `KeyError` if the element is not present.

```python
s = {10, 20, 30}
s.remove(20)  # {10, 30}
```

### 3. `clear()`

Removes all elements from the set.

```python
s = {10, 20, 30}
s.clear()  # set()
```

---

## Membership Operators

You can use the `in` and `not in` operators to check if an element exists in a set:

```python
s = {1, 2, 3, 'daniel'}
print(1 in s)  # True
print('z' in s)  # False
```

---

## Set Comprehension

Set comprehension allows you to create a new set from an iterable in a concise way:

```python
s = {x*x for x in range(5)}  # {0, 1, 4, 9, 16}
print(s)
```

---

## Removing Duplicates from a List

You can remove duplicate elements from a list by converting it into a set:

```python
a = [10, 20, 30, 10, 20, 40]
s = set(a)  # {10, 20, 30, 40}
print(s)
```

---

## Summary

| Feature               | Set                          |
|-----------------------|------------------------------|
| **Syntax**            | `{}` or `set()`              |
| **Mutability**        | Mutable                      |
| **Duplicates**        | Not allowed                  |
| **Order**             | Unordered                    |
| **Indexing**          | Not supported                |

- **When to use sets?**
  - When you need to store a collection of unique values and don’t care about order.
  - When you need fast membership testing or removing duplicates from a list.

---


# Dictionary Data Structure in Python

## Introduction

A **dictionary** in Python is a collection of key-value pairs. It is a mutable and unordered data structure, and it allows you to efficiently store and retrieve data based on unique keys.

### Key Features of Dictionaries

- **Key-Value Pairs:** Each item is a pair consisting of a key and its corresponding value.
- **No Duplicate Keys:** Dictionary keys must be unique. Duplicate keys are not allowed, but duplicate values can be stored.
- **Order:** Dictionaries are unordered, meaning the insertion order is not guaranteed (prior to Python 3.7).
- **Mutable:** Dictionaries are mutable, meaning their contents can be changed after creation.
- **No Indexing:** Elements are not stored in index order, and indexing or slicing doesn’t apply.

### When to Use a Dictionary?

Use a dictionary when you need to represent a collection of related data as key-value pairs. For example, you might use a dictionary to store an employee's name and their employee ID.

---

## Creating a Dictionary

A dictionary can be created using curly braces `{}` or the `dict()` function.

### Syntax

```python
# Creating a dictionary with key-value pairs
d = {key1: value1, key2: value2}
```

### Example

```python
# Creating a dictionary
d = {10: "Ramesh", 20: "Arjun", 30: "Daniel"}
print(d)  # Output: {10: 'Ramesh', 20: 'Arjun', 30: 'Daniel'}
```

### Empty Dictionary

```python
# Creating an empty dictionary
d = {}
print(d)  # Output: {}
```

You can add key-value pairs to the empty dictionary like this:

```python
# Adding key-value pairs to the empty dictionary
d[10] = "Ramesh"
d[20] = "Arjun"
d[30] = "Daniel"
```

---

## Accessing Values by Key

To access values in a dictionary, use their corresponding keys:

```python
# Accessing dictionary values by key
d = {10: "Ramesh", 20: "Arjun", 30: "Daniel"}
print(d[10])  # Output: Ramesh
print(d[20])  # Output: Arjun
print(d[30])  # Output: Daniel
```

You can also iterate over the dictionary using a loop:

```python
# Iterating through the dictionary
for k in d:
    print(k, d[k])
```

---

## Updating a Dictionary

You can update a dictionary by assigning a new value to an existing key or by adding a new key-value pair.

### Syntax

```python
d[key] = value  # To add or update a key-value pair
```

### Examples

1. **Adding a New Key-Value Pair**

```python
d = {10: "Ramesh", 20: "Arjun", 30: "Daniel"}
d[99] = "John"  # Adds new key-value pair
print(d)  # Output: {10: 'Ramesh', 20: 'Arjun', 30: 'Daniel', 99: 'John'}
```

2. **Updating an Existing Key-Value Pair**

```python
d = {10: 'Ramesh', 20: 'Arjun', 30: 'Daniel'}
d[30] = 'Chandhu'  # Updates value for key 30
print(d)  # Output: {10: 'Ramesh', 20: 'Arjun', 30: 'Chandhu'}
```

---

## Removing Elements from a Dictionary

You can remove dictionary elements using the `del` keyword or the `clear()` method.

1. **Using `del` Keyword**

```python
# Deleting a specific key-value pair
d = {10: "Ramesh", 20: "Arjun", 30: "Daniel"}
del d[10]  # Removes the entry with key 10
print(d)  # Output: {20: 'Arjun', 30: 'Daniel'}
```

2. **Using `clear()` Method**

```python
# Removing all elements from the dictionary
d = {10: "Ramesh", 20: "Arjun", 30: "Daniel"}
d.clear()  # Empties the dictionary
print(d)  # Output: {}
```

3. **Deleting the Entire Dictionary**

```python
# Deleting the entire dictionary object
del d
# Now, d is no longer accessible.
```

---

## Finding the Length of a Dictionary

You can find the number of items in a dictionary using the `len()` function:

```python
# Finding the length of a dictionary
d = {100: "Ramesh", 200: "Arjun"}
print(len(d))  # Output: 2
```

---

## Important Methods in Dictionary

Dictionaries have several useful methods that you can use to manipulate or query the data.

### 1. `clear()`

Removes all entries from the dictionary.

```python
d = {10: "Ramesh", 20: "Arjun", 30: "Daniel"}
d.clear()  # Empties the dictionary
```

### 2. `keys()`

Returns all keys in the dictionary.

```python
d = {100: "Ramesh", 200: "Arjun", 300: "Daniel"}
print(d.keys())  # Output: dict_keys([100, 200, 300])
```

### 3. `values()`

Returns all values in the dictionary.

```python
d = {100: "Ramesh", 200: "Arjun", 300: "Daniel"}
print(d.values())  # Output: dict_values(['Ramesh', 'Arjun', 'Daniel'])
```

### 4. `items()`

Returns all key-value pairs in the dictionary.

```python
d = {10: "Ramesh", 20: "Arjun", 30: "Daniel"}
for k, v in d.items():
    print(k, v)
```

---

## Dictionary Comprehension

Dictionary comprehension provides a concise way to create a dictionary from an iterable object (list, set, tuple, etc.).

```python
# Creating a dictionary of squares using comprehension
squares = {a: a*a for a in range(1, 6)}
print(squares)  # Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```

---

## Summary

| Feature               | Dictionary                    |
|-----------------------|-------------------------------|
| **Syntax**            | `{key: value, ...}`           |
| **Mutability**        | Mutable                       |
| **No Duplicates**     | Keys must be unique           |
| **Order**             | Unordered                     |
| **Indexing**          | Not supported                 |
| **Use Case**          | Storing key-value pairs       |

- **When to use dictionaries?**
  - When you need to store data as key-value pairs.
  - When you need fast lookups by key.
  - When you want to store a collection of unique items with associated values.

---


# Object-Oriented Programming (OOP) in Python

## Does Python follow a Functional or Object-Oriented approach?
- Python supports both functional and object-oriented programming.

## Features of Object-Oriented Programming (OOP)
- Class
- Object
- Constructor
- Inheritance
- And more…

---

## Class

### What is a Class?
- A **class** is a blueprint for creating objects (instances). It does not exist physically.
- A **class** defines the properties (attributes) and behaviors (methods) that the objects created from it will have.

### Syntax to Define a Class
```python
class ClassName:
    def __init__(self):
        # constructor
        self.property = value
    def method(self):
        # method or behavior
        pass
```

- `class` keyword is used to define a class.
- It can contain:
  - **Constructor**: Used for initialization.
  - **Properties**: Represent data (instance variables).
  - **Methods**: Represent actions or behavior.

### Example of a Simple Class
```python
class Employee:
    def display(self): 
        print("Hello, my name is Daniel")
```
- In the above example, `Employee` is a class with one method `display(self)`.

---

## Object

### What is an Object?
- An **object** is an instance of a class. It exists physically, while the class is a blueprint.
- An object holds data for the class's attributes and allows access to its methods.

### Why Create an Object?
- Objects store data for variables and methods defined in a class.
- They help allocate memory space for instance variables.

### Syntax to Create an Object
```python
object_name = ClassName()
```

### Example of Creating and Using an Object
```python
class Employee:
    def display(self): 
        print("Hello, my name is Daniel")

emp = Employee()  # Creating object
emp.display()  # Calling the method
```

---

## Constructor

### What is a Constructor?
- A **constructor** is a special method in Python used for initializing instance variables when an object is created.
- The constructor method name should be `__init__(self)`.

### When is the Constructor Executed?
- The constructor is automatically executed when an object of the class is created.

### Syntax of a Constructor
```python
class ClassName:
    def __init__(self):
        # Constructor body
        pass
```

### Example of Constructor
```python
class Employee:
    def __init__(self):
        print("Constructor is executed")

emp1 = Employee()  # Constructor is executed
```

### Types of Constructors
1. **Constructor without parameters**:
   - Used when no additional data is needed for initialization.
   - Syntax:
     ```python
     class ClassName:
         def __init__(self):
             pass
     ```

2. **Parameterized Constructor**:
   - Allows passing parameters during object creation to initialize instance variables.
   - Syntax:
     ```python
     class ClassName:
         def __init__(self, param1, param2):
             self.param1 = param1
             self.param2 = param2
     ```

### Example of Parameterized Constructor
```python
class Employee:
    def __init__(self, number):
        self.number = number
        print("Employee ID is:", self.number)

e1 = Employee(1)
e2 = Employee(2)
```

---

## Instance Variables

### What are Instance Variables?
- Instance variables are attributes that belong to an instance (object) of a class.
- Their values can change from one object to another.

### Example of Instance Variables
```python
class Student:
    def __init__(self, name, number):
        self.name = name
        self.number = number

s1 = Student("Daniel", 101)
print(s1.name)  # Output: Daniel
print(s1.number)  # Output: 101
```

---

## Instance Methods

### What are Instance Methods?
- Instance methods act upon the instance variables of a class.
- These methods are bound to an object and can access instance variables.

### Example of Instance Method
```python
class Demo:
    def __init__(self, a):
        self.a = a
    
    def m(self):
        print(self.a)

d = Demo(10)
d.m()  # Output: 10
```

---

## `self` in Python

### What is `self`?
- `self` is a predefined variable in Python that refers to the current instance of the class.
- It is used to initialize instance variables and create instance methods.

### Usage of `self`:
- **In the Constructor**: To initialize instance variables.
- **In Instance Methods**: To refer to instance variables and call methods.

---

## Difference Between Methods and Constructors

| Method                              | Constructor                             |
|-------------------------------------|-----------------------------------------|
| Used to perform actions or operations | Used to initialize instance variables |
| Method name can be anything         | The name must be `__init__(self)`      |
| Must be explicitly called           | Automatically called when an object is created |

---

## Conclusion

- Object-Oriented Programming (OOP) is a powerful paradigm in Python.
- By defining classes and creating objects, we can model real-world entities and organize code effectively.
- Key concepts like constructors, instance variables, and methods are essential for designing OOP systems in Python.

---


# Common Python Errors

### 1. SyntaxError
A **SyntaxError** occurs when there is a mistake in the syntax of your code, such as missing parentheses `()`, colons `:`, quotes `"`, or commas `,`.

**Example:**
```python
print("Hello World
```
*Explanation*: The error happens because the closing quotation mark is missing.

---

### 2. NameError
A **NameError** indicates that you are trying to use a variable that hasn't been defined yet.

**Example:**
```python
a = 10, 20, 30
print(b)
```
*Explanation*: Variable `b` is not defined, so Python raises a NameError.

---

### 3. IndexError
An **IndexError** happens when you try to access an index in a sequence (like a list or tuple) that is out of range.

**Example:**
```python
a = (10, 20, 30, 40)
print("Value: ", a[6])  # a[6] is out of range
```
*Explanation*: The index `6` does not exist in the tuple `a`.

---

### 4. TypeError
A **TypeError** occurs when an operation or function is applied to an object of inappropriate type.

**Example 1:**
```python
a = "hello"
b = 5
print(a + b)  # TypeError: cannot add string and integer
```

**Example 2:**
```python
a = (10, 20, 30)  # tuple is immutable
a[1] = 50  # TypeError: cannot modify a tuple
```
*Explanation*: In the first example, trying to add a string and an integer causes a TypeError. In the second example, trying to modify a tuple, which is immutable, raises an error.

---

### 5. ValueError
A **ValueError** occurs when an operation receives an argument with the right type, but an inappropriate value.

**Example:**
```python
a = "subh"
print(int(a))  # ValueError: invalid literal for int()
```
*Explanation*: The string `"subh"` cannot be converted to an integer.

---

### 6. KeyError
A **KeyError** happens when you try to access a key in a dictionary that does not exist.

**Example:**
```python
d = {10: "Ramesh", 20: "Subh", 30: "Daniel"}
del d[40]  # KeyError: 40 not in dictionary
```
*Explanation*: The key `40` is not present in the dictionary `d`.

---

### 7. IndentationError
An **IndentationError** occurs when the code is not properly indented. Python requires consistent indentation to define code blocks.

**Example 1:**
```python
a = [10, 20, 30, 40, 50]
for i in a:
print(i)  # IndentationError: expected an indented block
```

**Example 2:**
```python
class Employees:
    def display(self):
    print("Hello")  # IndentationError: expected an indented block
```
*Explanation*: The indentation of the `print()` statement is incorrect, leading to an error.

---

### 8. AttributeError
An **AttributeError** is raised when an attribute (method or variable) does not exist in an object.

**Example 1:**
```python
l = (10, 20, 30)
l.append(40)  # AttributeError: 'tuple' object has no attribute 'append'
```

**Example 2:**
```python
class Student:
    def subh(self):
        print("Hello")
s = Student()
s.m1()  # AttributeError: 'Student' object has no attribute 'm1'
```
*Explanation*: In the first example, the `tuple` object doesn't have the `append()` method. In the second example, there is no method `m1()` in the `Student` class.

---

### 9. ModuleNotFoundError
A **ModuleNotFoundError** occurs when you try to import a module that does not exist or can't be found.

**Example:**
```python
import keywords  # ModuleNotFoundError: No module named 'keywords'
```
*Explanation*: The module `keywords` doesn't exist.

---

### 10. TabError
A **TabError** happens when there is inconsistent use of tabs and spaces for indentation.

**Example:**
```python
def m1():
    for i in range(10):  # One space before 'for'
        print(i)          # Two spaces before 'print'
            return i      # One tab before 'return'
m1()
```
*Explanation*: Mixing spaces and tabs in indentation can cause this error.

---

### 11. TypeError: Unexpected Keyword Argument
This error occurs when a function is called with an unexpected keyword argument.

**Example 1:**
```python
def shopping(prod):
    print("Product is:", prod)

shopping(prod="iPhone 14 Plus")  # Correct usage
```

**Example 2:**
```python
def shopping(prod):
    print("Product is:", prod)

shopping(a="iPhone 14 Plus")  # TypeError: unexpected keyword argument 'a'
```
*Explanation*: In the second example, the keyword argument `a` does not match the function parameter `prod`, resulting in a TypeError.

