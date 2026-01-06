


# Functions in Python

---

## What is a Function?

A function is a block of reusable code that performs a specific task. It takes inputs (optional), processes them, and returns an output (optional). Functions help avoid repetition, making your code cleaner and more efficient.

### Why Use Functions?
- **Reusability**: Write code once and use it many times.
- **Maintainability**: Fix or update small parts without affecting the whole program.
- **Modularity**: Split big tasks into smaller, manageable pieces.
- **Readability**: Clear names and structure make code understandable.

In Machine Learning (ML) and Deep Learning (DL), functions are key for steps like cleaning data, training models, or calculating accuracy.

---

## Types of Functions

Python supports:
- **Built-in functions** (e.g., `print()`, `len()`)
- **User-defined functions** (created with `def`)
- **Anonymous functions** (using `lambda`)

### 1. Predefined Functions
These are built-in Python functions you can use right away.

#### Examples:
- `print()`: Shows text or numbers on the screen.
- `len()`: Tells you how many items are in a list or string.
- `type()`: Checks what kind of data something is (e.g., number, text).

In ML/DL, you might use `numpy.mean()` to find averages or `pandas.read_csv()` to load data.

### 2. User-Defined Functions
These are functions you create for your own needs.
These are functions created by developers to meet specific requirements.

#### Example:
```python
def clean_data(data):
    # Code to remove errors from data
    pass
```

---

## **Function Terminology**
To understand functions better, familiarize yourself with these terms:

- **`def` keyword:** Used to define a function.
- **Function Name:** The name of the function (use lowercase with underscores).
- **Parentheses `()`**: Used to pass parameters.
- **Parameters:** Variables that hold values passed into the function.
- **Colon `:`**: Marks the start of the function body.
- **Function Body:** The block of code that performs the task.
- **Return Type:** The value returned by the function (optional).
- **Indentation:** Required for defining the function body.

## How to Define a Function
Basic Steps
1. **Define it**: Tell Python what the function does.
2. **Call it**: Run the function when you need it.

### **1. Defining a Function**

Use the `def` keyword to define a function, followed by the function name and parentheses `()`. Parameters (inputs) are optional, and the function body is indented.

### Syntax
```python
def function_name(parameters):
    """Explain what the function does"""
    # Code goes here
    return result  # Optional
```

- **`def`**: Keyword to start a function.
- **Function Name**: Use lowercase with underscores (e.g., `train_model`).
- **Parameters**: Inputs the function uses (optional).
- **Return**: Sends a result back (optional).

#### Example:
```python
def add_numbers(a, b):
    """Adds two numbers"""
    return a + b

result = add_numbers(5, 3)  # Calling the function
print(result)  # Output: 8
```

> **Note:** When the function is defined, its body does not execute until the function is called.

---

## **Multiple Functions in a Program**
You can define and call multiple functions in a single program.

#### **Example:**
```python
def first():
    print("This is the first function")

def second():
    print("This is the second function")

first()  # Calling the first function
second()  # Calling the second function
```

---

## Functions Working Together

Functions can call other functions to share work.

#### Example:
```python
def preprocess_data(data):
    # Clean the data
    return cleaned_data

def analyze_data(data):
    cleaned = preprocess_data(data)
    # Analyze the cleaned data
    return result

analyze_data(raw_data)
```

This is common in ML: one function cleans data, another builds a model.

---

## **Types of Functions Based on Parameters**
Functions can be divided into two types based on parameters:

### 1. No Parameters
These don’t take any inputs.
A function that does not take any parameters.

#### Example:
```python
def start_training():
    print("Training started!")
start_training()
```

### 2. With Parameters
A function that accepts parameters.
These use inputs to do their job.

#### Example:
```python
def train_model(model, epochs):
    print(f"Training {model} for {epochs} rounds")
train_model("Neural Network", 10)
```

---

## Using `return`

### 1. No Return
Does something but doesn’t give back a value.

#### Example:
```python
def show_message():
    print("Task complete")
show_message()  # Output: Task complete
```

### 2. With Return
Gives back a value to use later.
**3. Return Statement**
- Functions can return values using `return`.
- If omitted, the function returns `None` by default.

#### Example:
```python
def get_accuracy(true, predicted):
    accuracy = # Calculate accuracy here
    return accuracy

score = get_accuracy(y_true, y_pred)
print(f"Accuracy: {score}")
```

In ML, `return` is used to send back metrics like accuracy or loss.

### Returning Multiple Values
Use tuples to return multiple values.

#### Example:
```python
def get_scores(predictions, true_labels):
    accuracy = 0.95  # Example value
    error = 0.05     # Example value
    return accuracy, error

acc, err = get_scores(y_pred, y_true)
print(f"Accuracy: {acc}, Error: {err}")
```

## **Return vs None**

If a function does not return anything, it returns `None` by default.

#### **Example:**
```python
def m1():
    print("This function returns nothing")

x = m1()
print(x)  # Output: None
```
---

## Arguments: Formal vs. Actual

- **Formal Arguments**: Names in the function definition (e.g., `model`, `epochs`).
- **Actual Arguments**: Values you pass when calling (e.g., `"CNN"`, `10`).

#### Example:
```python
def build_model(model_type, layers):  # Formal: model_type, layers
    print(f"Building {model_type} with {layers} layers")

build_model("CNN", 3)  # Actual: "CNN", 3
```

---

## Types of Arguments
Python offers flexibility in how you pass arguments to functions.

### 1. Positional Arguments
Arguments are passed in the same order as the parameters in the function definition.

#### Example:
```python
def subtract(x, y):
    return x - y
print(subtract(10, 5))  # Output: 5
print(subtract(5, 10))  # Output: -5
```

### 2. Keyword Arguments
Specify arguments by parameter names, ignoring order.

#### Example:
```python
def predict(model, data):
    print(f"Using {model} on {data}")

predict(data="test_set", model="DL")  # Works either way
predict(model="DL", data="test_set")
```

### 3. Default Arguments
Set a fallback value if nothing is passed.
Provide default values for parameters; they become optional during the call.

#### Example:
```python
def train(model, epochs=5):
    print(f"Training {model} for {epochs} epochs")

train("RNN")      # Uses epochs=5
train("RNN", 10)  # Overrides with epochs=10
```

### 4. Variable-Length Arguments
Handle many inputs with `*args` or `**kwargs`.
Python allows functions to accept variable numbers of arguments.
- **`*args`**: Accepts any number of positional arguments as a tuple.
- **`**kwargs`**: Accepts any number of keyword arguments as a dictionary.

#### Example with `*args`:
```python
def add_features(model, *features):
    for f in features:
        model.add(f)
add_features(model, "color", "size", "shape")
```

In DL, this could add layers to a neural network.

#### Example with `**kwargs`:
```python
def set_settings(**options):
    for key, value in options.items():
        print(f"{key}: {value}")
set_settings(learning_rate=0.01, batch_size=32)
```

### **Combining All**
```python
def describe_person(name, age=18, *hobbies, **details):
    print(f"Name: {name}, Age: {age}")
    print("Hobbies:", ", ".join(hobbies))
    print("Details:", details)

describe_person("David", 22, "reading", "swimming", job="Engineer", country="USA")
# Output:
# Name: David, Age: 22
# Hobbies: reading, swimming
# Details: {'job': 'Engineer', 'country': 'USA'}
```

---

## Lambda Functions
A `lambda` function is a small, one-line function defined without a name, often used for short-term tasks.

A **lambda function** is a small, anonymous function that can take any number of arguments but only has one expression.

### Syntax:
```python
lambda arguments: expression
```

#### Example:
```python
double = lambda x: x * 2
print(double(4))  # Output: 8
```

In ML, lambdas help with quick data transformations.

---

## Lambda with Built-in Helpers
Lambda functions are often used with `map()`, `filter()`, and `reduce()`.

### 1. `map()`
Applies a function to every item in a list.

#### Example:
```python
data = [1, 2, 3]
squared = map(lambda x: x**2, data)
print(list(squared))  # Output: [1, 4, 9]
```

### 2. `filter()`
Keeps items that meet a condition.
Filters elements based on a condition.

#### Example:
```python
numbers = [1, 2, 3, 4]
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))  # Output: [2, 4]
```

### 3. `reduce()`
Combines items into one value.
Applies a function cumulatively to elements in an iterable.

#### Example:
```python
from functools import reduce
numbers = [1, 2, 3, 4]
total = reduce(lambda x, y: x + y, numbers)
print(total)  # Output: 10
```

In ML, these are great for processing data fast.

---

## Summary

- Functions make code **organized, reusable, and clear**.
- Use **predefined functions** for common tasks and **user-defined functions** for custom ones.
- Arguments can be **positional, keyword, default, or variable-length**.
- **Lambda functions** are quick helpers for simple jobs.
- In ML/DL, functions handle everything from data prep to model evaluation.

--- 

## **6. Recursion**
A function can call itself to solve problems by breaking them into smaller instances.

### **Example: Factorial**
```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # Output: 120 (5 * 4 * 3 * 2 * 1)
```

### **Fibonacci Sequence**
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(6))  # Output: 8 (0, 1, 1, 2, 3, 5, 8)
```

---
## **Decorators**
Decorators are functions that modify the behavior of other functions. They’re often used for logging, timing, or access control.

### **Syntax**
```python
def decorator(func):
    def wrapper():
        print("Before the function")
        func()
        print("After the function")
    return wrapper

@decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Before the function
# Hello!
# After the function
```

### **With Arguments**
```python
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def cheer(name):
    print(f"Go, {name}!")

cheer("Team")
# Output:
# Go, Team!
# Go, Team!
# Go, Team!
```

---

## **8. Function Annotations**
Annotations provide metadata about parameter and return types (not enforced).

```python
def add(a: int, b: int) -> int:
    return a + b

print(add(3, 4))  # Output: 7
print(add.__annotations__)  # Output: {'a': <class 'int'>, 'b': <class 'int'>, 'return': <class 'int'>}
```

---
## **Common Tools and Methods**
- **`callable()`**: Checks if an object is callable (e.g., a function).
  ```python
  def test(): pass
  print(callable(test))  # Output: True
  print(callable(5))     # Output: False
  ```
- **`__doc__`**: Access the docstring.
  ```python
  print(calculator.__doc__)  # Output: A flexible calculator...
  ```
- **`partial` from functools`**: Fixes some arguments of a function.
  ```python
  from functools import partial
  def power(base, exp):
      return base ** exp
  square = partial(power, exp=2)
  print(square(3))  # Output: 9
  ```

---

## **Practical Example: All-in-One Calculator**
Let’s combine everything into a versatile calculator function.

```python
def calculator(operation: str, *args, precision: int = 2, **kwargs) -> float:
    """A flexible calculator with variable arguments and precision control."""
    def log_result(func):
        def wrapper(*a, **kw):
            result = func(*a, **kw)
            print(f"Operation '{operation}' result: {result}")
            return result
        return wrapper
    
    @log_result
    def compute():
        if operation == "add":
            return round(sum(args), precision)
        elif operation == "multiply":
            result = 1
            for num in args:
                result *= num
            return round(result, precision)
        elif operation == "power":
            base = kwargs.get("base", args[0])
            exponent = kwargs.get("exponent", args[1] if len(args) > 1 else 2)
            return round(base ** exponent, precision)
        else:
            raise ValueError("Unsupported operation")

    return compute()

# Usage
print(calculator("add", 1.234, 2.345, 3.456))  # Output: Operation 'add' result: 7.04
print(calculator("multiply", 2, 3, 4))         # Output: Operation 'multiply' result: 24
print(calculator("power", base=2, exponent=3))  # Output: Operation 'power' result: 8
```


