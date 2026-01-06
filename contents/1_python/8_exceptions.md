
---

# Exception Handling

---

## Types of Errors in Programming

In Python, errors fall into two categories:

- **Syntax Errors**: Mistakes in code structure that prevent the program from running. Examples include missing punctuation like colons or parentheses.
  ```python
  # Syntax Error: missing colon
  x = 123
  if x == 123  # Missing ':'
      print("Hello")
  ```
  - **Fix**: Add the colon: `if x == 123:`
  - **Responsibility**: Programmers must correct these before execution.

- **Runtime Errors (Exceptions)**: Problems that occur while the program runs, such as dividing by zero or accessing a missing file. These are manageable with exception handling.
  ```python
  # Runtime Error: division by zero
  print(10 / 0)  # Raises ZeroDivisionError
  ```
  - These are common in ML/DL when handling datasets or computations.

Exception handling focuses on runtime errors, not syntax errors, to keep programs running smoothly.

---

## What is an Exception?

An exception is an unexpected issue that disrupts a program’s normal execution. If unhandled, it causes the program to stop. Common examples include:

- `ZeroDivisionError`: Attempting to divide by zero.
- `ValueError`: Converting invalid data, e.g., `int("text")`.
- `FileNotFoundError`: Opening a nonexistent file, frequent in ML when loading datasets.

### Why Exception Handling Matters in ML/DL
In ML and DL, exceptions can:
- Halt data preprocessing or model training unexpectedly.
- Lock resources like files or GPU memory, affecting system performance.

Handling exceptions ensures programs terminate gracefully and resources are freed.

---

## Normal vs. Abnormal Program Flow

- **Normal Flow**: The program executes all steps successfully.
  ```python
  print("Loading data...")
  print("Training model...")
  print("Done!")
  ```
  - Output: All lines print without interruption.

- **Abnormal Flow**: An exception stops execution midway.
  ```python
  print("Loading data...")
  print(10 / 0)  # Raises ZeroDivisionError
  print("Training model...")  # Never reached
  ```
  - Output: Stops at the error, skipping later steps.

In ML/DL, abnormal flow can disrupt pipelines. Exception handling provides a backup plan.

---

## Python’s Default Exception Handling

When an exception occurs, Python:
1. Creates an exception object.
2. Looks for handling code.
3. If none is found, it stops the program and displays the error.
   ```python
   print("Start")
   print(10 / 0)  # No handling code
   print("End")
   ```
   - Output: `ZeroDivisionError: division by zero`, and "End" is skipped.

In ML/DL, this default behavior can leave projects incomplete. Custom handling is key.

---

## Handling Exceptions with Try-Except

The `try-except` structure manages runtime errors:

- **Try Block**: Contains code that might fail, like loading data or training a model.
- **Except Block**: Runs if an error occurs, providing an alternative action.

### Example: Loading Data in ML
```python
try:
    data = open("dataset.csv").read()  # Might fail if file is missing
except FileNotFoundError:
    print("Dataset not found. Using default data.")
    data = "default,data"
print("Processing data...")
```
- **Success**: File loads, except block is skipped.
- **Failure**: File is missing, except block provides default data.

This ensures ML pipelines continue despite missing files.

### Control Flow
- **No Error**: Try runs, except is skipped.
- **Error Caught**: Try stops at the error, except handles it, and the program continues.
- **Error Not Caught**: If the except block doesn’t match the error type, the program stops.

---

## Using Multiple Except Blocks

Handle different errors with separate `except` blocks:
```python
try:
    x = int(input("Enter a number: "))
    result = 10 / x
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Please enter a valid number!")
```
- Input `0`: Triggers `ZeroDivisionError`.
- Input `"text"`: Triggers `ValueError`.

In ML, this helps manage diverse errors, like invalid data or computational issues.

---

## The Finally Block for Cleanup

The `finally` block runs always, whether an exception occurs or not. It’s ideal for cleanup tasks, such as closing files or freeing memory.

### Example: Resource Management in DL
```python
try:
    model = load_model("model.h5")  # Might fail
except FileNotFoundError:
    print("Model file not found!")
finally:
    print("Releasing resources...")
    # Free GPU memory or close files
```
- **Purpose**: Prevents resource leaks in long-running ML/DL tasks.

---

## Nested Try-Except Blocks in ML Pipelines

Complex ML workflows may need nested handling:
```python
try:
    # Outer: Model training
    try:
        # Inner: Data loading
        data = load_data("data.csv")  # Might raise FileNotFoundError
    except FileNotFoundError:
        print("Data file missing, using backup.")
        data = "backup,data"
    train_model(data)
except Exception as e:
    print(f"Training failed: {e}")
finally:
    print("Cleaning up...")
```
- Inner block handles data errors.
- Outer block catches training issues.
- Finally ensures cleanup.

This keeps ML pipelines robust across multiple stages.

---

## The Else Block

The `else` block runs only if no exception occurs in the `try` block:
```python
try:
    data = load_data("data.csv")
except FileNotFoundError:
    print("Data file not found!")
else:
    print("Data loaded successfully!")
finally:
    print("Cleanup complete.")
```
- In ML, use `else` to proceed with training only if data loading succeeds.

---

## Common Exceptions in ML/DL

- **Predefined Exceptions**:
  - `MemoryError`: Running out of memory with large datasets.
  - `IndexError`: Accessing invalid indices in NumPy arrays.

- **Custom Exceptions**: Define your own for specific needs.
  ```python
  class EmptyDatasetError(Exception):
      pass

  def check_data(data):
      if not data:
          raise EmptyDatasetError("Dataset is empty!")
  ```
  - Useful for ML-specific issues, like invalid model inputs.

---

## Practical Example: Validating Data with Regular Expressions

Regular expressions (regex) help validate data, reducing runtime errors in ML/DL.

### Example: Checking Email Formats
```python
import re

email = "user@domain.com"
if re.fullmatch(r"\w+@\w+\.\w+", email):
    print("Valid email")
else:
    print("Invalid email")
```
- Ensures clean data inputs for ML models.

---

## Conclusion

Exception handling in Python is essential for ML and DL, where data and resource challenges are frequent. Using `try-except`, `finally`, and nested blocks, you can:
- Manage errors without crashing.
- Release resources reliably.
- Keep workflows intact.

---
