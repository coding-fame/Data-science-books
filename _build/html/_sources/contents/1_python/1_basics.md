
---

# Python Basics

---

## 1. Introduction

### What is Python?
- **Python** is a general purpose and high-level programming language.
- **General purpose** means that many companies use Python to develop, test, and deploy software applications.
- **High-level programming** means it’s human-readable and easy to understand.

- **Python** is a popular, high-level programming language that’s easy to read and write.
- **High-level** means it’s designed for humans to understand, not just machines.
- It’s widely used in many fields, especially Machine Learning (ML) and Deep Learning (DL).

### Why Use Python for Machine Learning?
Python is perfect for ML because:
- It’s simple to learn and use.
- It has powerful libraries like **NumPy** (for math), **Pandas** (for data handling), **Scikit-learn** (for ML models), and **TensorFlow** (for deep learning).
- These tools help you process data, build models, and train them efficiently.

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

### History of Python
- Created by Guido van Rossum in 1991.
- It’s **open-source**, meaning it’s free to use and download from [www.python.org](https://www.python.org).
- Supports **functional programming** (using functions) and **object-oriented programming** (using classes), which are great for organizing ML projects.

---

## 2. Python Keywords

Keywords are special words in Python with specific meanings. You can’t use them as variable names. They’re important for writing ML code correctly.

Here’s a table of common Python keywords:

| **Category**         | **Keywords**                           |
|-----------------------|----------------------------------------|
| **Control Flow**     | `if`, `else`, `elif`, `for`, `while`, `break`, `continue`, `pass` |
| **Functions**        | `def`, `lambda`, `return`, `yield`    |
| **Classes**          | `class`, `with`                       |
| **Modules**          | `import`, `from`                      |
| **Logic**            | `and`, `or`, `not`                    |
| **Constants**        | `True`, `False`, `None`               |
| **Exceptions**       | `try`, `except`, `finally`, `raise`   |

**Key Points:**
- Keywords are lowercase, except `True`, `False`, and `None`.
- Example in ML: Use `if` and `else` to check model accuracy during training.

---

## 3. Your First Python Program

Let’s write a simple program, like one you might use to test an ML idea.

### How to Write and Run It
You can use:
- Text editors (e.g., Notepad++, VS Code).
- IDEs (e.g., Jupyter Notebook, PyCharm) – great for ML because they show output instantly.

**Example: Hello World**
```python
print("Hello World")
```
- `print()` shows text or results on the screen.
- `"Hello World"` is a **string** (text data).

**Steps:**
1. Write the code in a file, e.g., `hello.py`.
2. Run it with:  
   ```bash
   python hello.py
   ```
3. See the output: `Hello World`.

---

## 4. How Python Code Runs

Knowing how Python works helps when running big ML models.

### Execution Steps
1. **Source Code**: Your Python file (e.g., `model.py`).
2. **Byte Code**: Python converts it to a faster format (stored as `.pyc` files).
3. **Python Virtual Machine (PVM)**: Runs the byte code to give you results.

**Why It Matters for ML:**
- Python is **interpreted**, so it runs line by line. This makes debugging ML code easier.

---

## 5. Naming Conventions

Good names make your ML code easy to read and share.

### What Are Identifiers?
Identifiers are names for things like variables, functions, or classes.

**Rules:**
- Use letters, numbers, and underscores (`_`).
- Don’t start with a number. `SyntaxError`
- They’re case-sensitive (e.g., `Data` ≠ `data`). `NameError`
- Don’t use keywords (e.g., `if`). `SyntaxError`
- Don't use spaces (e.g. `a b`). `SyntaxError`

**Best Practices for ML:**
- **Variables**: Use `snake_case` (e.g., `training_data`).
- **Functions**: Use `snake_case` (e.g., `predict_output`).
- **Classes**: Use `PascalCase` (e.g., `NeuralNetwork`).
- **Constants**: Use `UPPER_CASE` (e.g., `LEARNING_RATE`).

**Example:**
```python
learning_rate = 0.01  # Good naming for an ML parameter
```

---

## Comments in a Program

1. **Single-line Comments:**  
   Use the `#` symbol.
2. **Multi-line Comments:**  
   Enclose the comment between triple quotes (`''' ... '''` or `""" ... """`).

---

## 6. Variables
A **variable** is a reserved memory location used to store values.
Variables store data, like numbers or model results, for your program to use.

### Basics
- **Name**: What you call it (e.g., `score`).
- **Value**: What it holds (e.g., `95`).
- **Type**: What kind of data it is (e.g., number).

**Example:**
```python
score = 95  # Variable storing a model’s accuracy
```

## Variable Re-initialization:
- Re-initializing a variable replaces its old value with a new one.

**In ML:**
- Variables might store training data, weights, or predictions.

---

## 7. Data Types
A **data type** represents the type of data stored in a variable or memory.

Data types tell Python what kind of data you’re working with. This is critical in ML for efficiency.

### Common Types
- **int**: Whole numbers (e.g., `10` – epochs in training).
- **float**: Decimals (e.g., `0.85` – accuracy score).
- **bool**: `True` or `False` (e.g., `is_trained`).
- **str**: Text (e.g., `"model"`).
- **list**: Changeable list (e.g., `[1, 2, 3]` – data points).

**Check Type:**
```python
x = 3.14
print(type(x))  # Output: <class 'float'>
```

**ML Tip:**
- Use `float` for model parameters like weights to allow precise calculations.

---

## 8. Python Operators
An **operator** is a symbol that performs an operation on one or more operands (variables).
Operators let you work with data, like doing math or checking conditions in ML.

> **Note:** Python does not have increment (++) and decrement (--) operators.

### 1. Arithmetic Operators
Arithmetic operators perform basic mathematical operations.

For calculations, like computing loss in ML.

- `+` : Add
- `-` : Subtract
- `*` : Multiply
- `/` : Divide (gives a float)
- `**` : Power (e.g., `2 ** 3` = 8)
- `%` : Modulus (remainder of division)
- `//` : Floor division (integer quotient)

**Example:**
```python
loss = 0.5 * error  # Calculate part of a loss function
```

### 2. Assignment Operators
Assignment operators are used to assign values to variables.
Update variables, like adjusting weights during training.

- `=` : Set a value
- `+=` : Add and set
- `-=` : Subtract and set

**Example:**
```python
weight = 0.1
weight += 0.05  # Now weight is 0.15
```

### 3. Unary Minus Operator
The unary minus operator is used to change the sign of a number.

- `-` : Unary minus operator.

### 4. Relational Operators
Compare values, like checking if a model improved.

- `>` : Greater than
- `<` : Less than
- `==` : Equal to

**Example:**
```python
if accuracy > 0.9:
    print("Great model!")
```

### 5. Logical Operators
Logical operators are used to combine conditional statements or create compound conditions.

Combine conditions, like stopping training early.

- `and` : Both must be true
- `or` : One must be true
- `not` : Flip the value

**Example:**
```python
if epochs > 10 and loss < 0.1:
    print("Stop training")
```

### 7. Membership Operators
Membership operators are used to test whether a value exists in a sequence (e.g., string, list, set, tuple, dictionary).

- `in` : Returns `True` if the element is found in the sequence.
- `not in` : Returns `True` if the element is not found in the sequence.

---

## Summary
- Python is key for ML because it’s simple and has great tools.
- Learn keywords, variables, and data types to build solid code.
- Use operators for math and logic in ML tasks.
- Follow naming rules to keep your code clear and professional.

