
---

# Data Structures

If you wanted to store a group of individual objects in a single entity, then you should go for data structures.

Data structures are essential tools in Python, especially for Machine Learning (ML) and Deep Learning (DL). They help organize, store, and process large datasets efficiently. 

---

## Why Learn Data Structures for ML?

Data structures are critical in ML and DL for tasks like:

- **Storing** and **retrieving** large datasets (e.g., training data).
- **Processing** data efficiently (e.g., feature extraction in NLP).
- **Handling duplicates** (e.g., removing redundant labels).
- **Maintaining order** (e.g., ordered predictions or sorted features).
- **Sorting** and **accessing data quickly** (e.g., sorted model outputs).
- **Managing key-value pairs** (e.g., storing model parameters).

Python’s built-in data structures, such as **strings**, **lists**, **tuples**, **sets**, and **dictionaries**, are versatile and widely used in ML workflows.

---

Python provides:
- **Built-in Data Structures**: Lists, Tuples, Dictionaries, Sets
- **Advanced Data Structures**: Available via modules like `collections`, `heapq`, `array`, etc.

## What is a Sequence?

A **sequence** is a data structure that holds a group of elements in a specific order. In Python, common sequences include:

- **Strings**: Text data (e.g., class labels in classification tasks).
- **Lists**: Mutable collections (e.g., feature vectors for ML models).
- **Tuples**: Immutable collections (e.g., fixed hyperparameters).
- **Sets**: Unique elements (e.g., unique categories in a dataset).
- **Dictionaries**: Key-value pairs (e.g., model configurations).

Sequences are essential in ML for organizing data, labels, and outputs efficiently.

---

# Working with Strings in Python

## What is a String?
### Definitions:
1. A string is a **group of characters** enclosed within single, double, or triple quotes.
2. A string is a **sequential collection of characters**.

A **string** is a sequence of characters enclosed in quotes. In ML, strings are used for text data, such as:

- Class labels (e.g., "cat", "dog" in image classification).
- Text features (e.g., customer reviews for sentiment analysis).
- File paths (e.g., locations of datasets or models).

### Example:
```python
label = "cat"  # String for a class label in ML
```

---

## Creating Strings

You can create strings using different types of quotes:

- Single quotes: `'Hello'`
- Double quotes: `"Hello"` (most common).
- Triple quotes: `'''Hello'''` or `"""Hello"""` (for multiline strings).

**ML Tip:** Use triple quotes for multiline text data, such as reading large text files for Natural Language Processing (NLP).

### Examples:
```python
# Single-line string
label = "dog"

# Multiline string for NLP tasks
text_data = """This is a sample review.
It spans multiple lines."""
```

---

## Indexing and Slicing Strings

### Indexing
Each character in a string has an index, starting from 0. Indexing is useful for accessing specific parts of text data.

**Example:**
```python
text = "Machine Learning"
print(text[0])  # Output: M
print(text[8])  # Output: L
```

### Slicing
Slicing extracts a substring using a range of indices. In ML, this can help extract features from text.

**Example:**
```python
text = "Machine Learning"
print(text[0:7])  # Output: Machine
```

### Looping Through Strings
Use a `for` loop to process each character. This is helpful for text preprocessing in NLP.

**Example:**
```python
for char in "ML":
    print(char)
# Output:
# M
# L
```

---

## Mutability and Immutability

- **Mutable**: Can be changed after creation (e.g., lists).
- **Immutable**: Cannot be changed after creation (e.g., strings).

Strings are **immutable**, meaning you cannot modify them directly. In ML, this ensures data integrity for text inputs.

**Example:**
```python
text = "ML"
# text[0] = "D"  # Error: 'str' object does not support item assignment
```

**ML Tip:** Since strings are immutable, use methods like `replace()` to create modified versions without altering the original.

---

## Mathematical Operations on Strings

### 1. Concatenation (`+`)
Joins two or more strings. Useful for combining text features or labels.

**Example:**
```python
model = "Logistic" + "Regression"
print(model)  # Output: LogisticRegression
```

### 2. Repetition (`*`)
Repeats a string multiple times. Can be used to create repeated labels or dummy data.

**Example:**
```python
label = "spam" * 3
print(label)  # Output: spamspamspam
```

---

## String Length

Use `len()` to find the number of characters in a string. In ML, this helps measure text lengths for padding or truncation in NLP.

**Example:**
```python
text = "Deep Learning"
print(len(text))  # Output: 13
```

---

## Membership Operators (`in`, `not in`)

### `in`
Checks if a substring exists within a string. Useful for filtering text data.

**Example:**
```python
print('Ma' in 'Machine Learning')  # Output: True
print('AI' in 'Machine Learning')   # Output: False
```

### `not in`
Checks if a substring does **not** exist. Helps exclude certain patterns.

**Example:**
```python
print('DL' not in 'Machine Learning')  # Output: True
```

---

## Key String Methods for ML

Strings come with many built-in methods. Here are essential ones for ML tasks:

| **Method**       | **Purpose**                                   | **Example**                            |
|------------------|-----------------------------------------------|----------------------------------------|
| `upper()`        | Converts to uppercase                         | `"ml".upper() → "ML"`                  |
| `lower()`        | Converts to lowercase                         | `"ML".lower() → "ml"`                  |
| `strip()`        | Removes leading/trailing spaces               | `"  ML  ".strip() → "ML"`              |
| `count(p)`       | Counts occurrences of `p`                     | `"ML ML".count("ML") → 2`              |
| `replace(p1, p2)`| Replaces `p1` with `p2`                       | `"ML".replace("M", "D") → "DL"`        |
| `split(p)`       | Splits string at separator `p` into a list    | `"ML,DL".split(",") → ["ML", "DL"]`    |

**ML Use Cases:**
- **`lower()`**: Standardize text data for case-insensitive processing.
- **`strip()`**: Clean extra spaces from user inputs or datasets.
- **`split()`**: Tokenize text into words for NLP models.

### Example: Text Preprocessing
```python
# Clean and tokenize text for NLP
text = "  Machine Learning is fun!  "
cleaned = text.strip().lower()
tokens = cleaned.split(" ")
print(tokens)  # Output: ['machine', 'learning', 'is', 'fun!']
```

---

## Immutability and String Methods

Though strings are immutable, methods like `replace()` return a **new string** with changes. This is useful for creating modified copies without altering the original data.

**Example:**
```python
original = "Neural Network"
modified = original.replace("Neural", "Convolutional")
print(modified)  # Output: Convolutional Network
print(original)  # Output: Neural Network (unchanged)
```

**ML Tip:** Use this to preprocess text while keeping the original intact for reference.

---

## Splitting Strings

The `split()` method divides a string into a list, which is crucial for tokenizing text in NLP.

- **Default (no separator)**: Splits at spaces.
- **With separator**: Splits at the specified character.

**Example:**
```python
# Tokenize text for ML
text = "Machine Learning, Deep Learning"
tokens = text.split(", ")  # Output: ['Machine Learning', 'Deep Learning']
```

**ML Tip:** Use `split()` to break down sentences into words or phrases for feature extraction.

---

## Summary

- **Strings** are immutable sequences of characters, ideal for handling text data in ML.
- Use **indexing**, **slicing**, and **looping** to access and process parts of strings.
- **String methods** like `lower()`, `strip()`, and `split()` are essential for preprocessing text data.
- **Membership operators** (`in`, `not in`) help filter and validate text inputs.
- Understanding strings is key for tasks like NLP, where text data is common.

---

# Python Lists for Machine Learning

A **list** is a mutable, ordered sequence of elements, enclosed in square brackets `[]`.

---

## What is a List?

- **How to Create a List:**
  - Use square brackets: `[]`
  - Use the `list()` function
- **Key Features:**
  - **Same or Different Types:** Lists can hold similar items (e.g., all numbers) or mixed items (e.g., numbers and text).
  - **Dynamic Size:** You can add or remove items anytime.
  - **Order Matters:** Items stay in the order you add them.
  - **Duplicates Allowed:** The same value can appear multiple times.
  - **Mutable:** You can update items after creating the list.
  - **Indexed:** Each item has a position (index) starting from 0.

**In ML:** Lists are great for storing datasets, model predictions, or hyperparameters.

Note:
- A list is a predefined class in Python.
- Once we create a list object, an internal object is created for the list class.

---

## Creating Lists

Here’s how to make lists with simple examples:

1. **Empty List**
   ```python
   data = []  # An empty list to store ML results later
   ```

2. **List with Items**
   ```python
   features = ["age", 25, 0.85]  # Mixed types: text, integer, float
   print(features)  # Output: ["age", 25, 0.85]
   ```

3. **Using `list()` Function**
    - `list(p)` is a predefined function in Python.
    - This function takes only one parameter.
    - Converts a sequence (like a range) into a list.
   
   ```python
   epochs = list(range(1, 6))  # Creates [1, 2, 3, 4, 5]
   print(epochs)
   ```

**ML Example:** You might use `list(range(10))` to create a list of training epochs.

---

## Lists Are Mutable

“Mutable” means you can change a list after creating it. This is handy in ML when updating values like weights.

```python
weights = [0.1, 0.2, 0.3]
print(weights)  # Output: [0.1, 0.2, 0.3]
weights[0] = 0.5  # Update first item
print(weights)  # Output: [0.5, 0.2, 0.3]
```

---

## Accessing List Items

You can grab items from a list in different ways:

1. **By Index**
   - Indexes start at 0 (left to right).
   - Negative indexes start from -1 (right to left).
   ```python
   scores = [0.9, 0.85, 0.95]
   print(scores[0])   # Output: 0.9 (first item)
   print(scores[-1])  # Output: 0.95 (last item)
   ```
   - **Warning:** Using an index beyond the list size (e.g., `scores[5]`) causes an `IndexError`.

2. **Slicing**
   - Extract a chunk of the list with `[start:stop:step]`.
    - `start`: Index where slice starts (default is 0)
    - `stop`: Index where slice ends (default is max index of list)
    - `stepsize`: Increment value (default is 1)
   ```python
   data = [10, 20, 30, 40, 50]
   print(data[1:4])  # Output: [20, 30, 40]
   print(data[::2])  # Output: [10, 30, 50] (every second item)
   ```

3. **Using a For Loop**
   - Loop through items one by one.
   ```python
   predictions = [0.7, 0.8, 0.9]
   for pred in predictions:
       print(pred)
   ```

**ML Tip:** Slicing is useful for splitting data into training and testing sets.

---

## Finding the Length of a List

Use `len()` to count items in a list.

```python
values = [10, 20, 30, 40]
print(len(values))  # Output: 4
```

**In ML:** Check the size of your dataset with `len(dataset)`.

---

## List Methods

Lists come with built-in methods to manage items. Here are some key ones:

- `append(x)`: Add an item to the end.
- `extend(iterable)`: Append all items from an iterable.
- `insert(i, x)`: Insert at index `i`.
- `remove(x)`: Remove first occurrence of `x`.
- `pop([i])`: Remove and return item at index `i` (default: last).
- `index(x)`: Return index of first `x`.
- `count(x)`: Count occurrences of `x`.
- `sort()`: Sort in-place.
- `reverse()`: Reverse in-place.

**Example:**
```python
scores = [0.9, 0.8, 0.9]
print(scores.count(0.9))  # Output: 2
scores.append(0.95)
print(scores)  # Output: [0.9, 0.8, 0.9, 0.95]
scores.sort()
print(scores)  # Output: [0.8, 0.9, 0.9, 0.95]
```
**ML Use:** Use `append()` to collect model accuracies during training.

```python
# Creating a list
fruits = ["apple", "banana", "cherry"]

# Adding elements
fruits.append("orange")
fruits.extend(["grape", "kiwi"])
print(fruits)  # Output: ['apple', 'banana', 'cherry', 'orange', 'grape', 'kiwi']

# Inserting and removing
fruits.insert(1, "mango")
fruits.remove("banana")
print(fruits.pop())  # Output: kiwi
print(fruits)  # Output: ['apple', 'mango', 'cherry', 'orange', 'grape']

# Sorting and reversing
numbers = [3, 1, 4, 1, 5]
numbers.sort()
print(numbers)  # Output: [1, 1, 3, 4, 5]
numbers.reverse()
print(numbers)  # Output: [5, 4, 3, 1, 1]
```

---

## List Operators

Operators let you combine or repeat lists.

1. **Concatenation (`+`)**
   - Joins two lists into one.
   ```python
   list1 = [1, 2]
   list2 = [3, 4]
   combined = list1 + list2
   print(combined)  # Output: [1, 2, 3, 4]
   ```

2. **Repetition (`*`)**
   - Repeats a list a set number of times.
   ```python
   data = [0] * 3
   print(data)  # Output: [0, 0, 0]
   ```

3. **Membership (`in`, `not in`)**
   - Checks if an item is in the list.
   ```python
   values = [10, 20, 30]
   print(20 in values)      # Output: True
   print(40 not in values)  # Output: True
   ```

**ML Example:** Use `+` to combine feature lists or `in` to check if a label exists.

---

## List Comprehension
List comprehension provides a concise way to create lists. 

List comprehension is a short, powerful way to create or modify lists. It’s like a one-line loop.

**Syntax:**
```python
new_list = [expression for item in iterable if condition]
```

**Examples:**
1. **Add 1 to Each Item**
   ```python
   losses = [0.5, 0.6, 0.7]
   new_losses = [loss + 1 for loss in losses]
   print(new_losses)  # Output: [1.5, 1.6, 1.7]
   ```

2. **Filter Values**
   ```python
   accuracies = [0.8, 0.9, 0.7, 0.95]
   high = [acc for acc in accuracies if acc >= 0.9]
   print(high)  # Output: [0.9, 0.95]
   ```

3. **Squares of Numbers**
   ```python
   numbers = range(1, 5)
   squares = [n ** 2 for n in numbers]
   print(squares)  # Output: [1, 4, 9, 16]
   ```

**ML Use:** Quickly process data or calculate metrics like squared errors.

---

## Summary

- **Lists** are flexible containers for storing data in Python.
- They’re **mutable**, so you can change them as needed.
- Use **indexes** or **slicing** to access items, and **methods** like `append()` or `sort()` to manage them.
- **Operators** (`+`, `*`) and **comprehensions** make lists even more powerful.
- In **Machine Learning**, lists help store datasets, predictions, or parameters.

--- 

# Tuple Data Structure in Python

A **tuple** is an immutable, ordered sequence of elements, enclosed in parentheses `()`.

## Introduction
A **tuple** is a built-in data structure in Python used to store a collection of elements. Unlike lists, tuples are **immutable**, meaning their contents cannot be changed after creation. This makes tuples ideal for storing fixed data that should remain constant throughout the program.

---

## Key Features of Tuples
1. **Order Preservation:** Tuples maintain the order of elements as they are inserted.
   - Example: `(10, 20, 30)` will always output as `(10, 20, 30)`.

2. **Duplicate Elements:** Tuples allow duplicate values.
   - Example: `(10, 20, 20)` is valid.

3. **Immutable:** Once created, tuples cannot be modified.
   - Example: You cannot add, remove, or change elements after creation.

4. **Indexed Access:** Elements in a tuple can be accessed using indices.
   - Positive indices start from the left (`t[0]`).
   - Negative indices start from the right (`t[-1]`).

---

## When to Use Tuples
Tuples are best suited for scenarios where data should remain constant, such as:
- **Days of the week**
- **Month names**
- **Configuration values**
- **Dictionary keys** (since they are immutable and hashable)

---

## Creating Tuples
Tuples can be created in several ways:

### 1. Using Parentheses
Parentheses `()` are commonly used, but they are optional.
```python
employee_ids = (10, 20, 30, 40, 50)
print(employee_ids)  # Output: (10, 20, 30, 40, 50)
```

### 2. Using the `tuple()` Function
Convert other iterables (like lists) into tuples.
```python
a = [11, 22, 33]
t = tuple(a)
print(t)  # Output: (11, 22, 33)
```

### 3. Single-Element Tuple
A single-element tuple must include a trailing comma.
```python
number = (9)  # Not a tuple
print(type(number))  # Output: <class 'int'>

name = ("Daniel",)  # Single-element tuple
print(type(name))  # Output: <class 'tuple'>
```

### 4. Without Parentheses
Parentheses are optional when creating tuples.
```python
emp_ids = 10, 20, 30, 40
print(emp_ids)  # Output: (10, 20, 30, 40)
```

---

## Accessing Tuple Elements
Tuples support indexing and slicing for accessing elements.

### 1. Indexing
Access elements using their position.
```python
t = (10, 20, 30, 40, 50, 60)
print(t[0])  # Output: 10
print(t[-1])  # Output: 60
```

### 2. Slicing
Extract a subset of elements using slicing.
```python
t = (10, 20, 30, 40, 50, 60)
print(t[2:5])  # Output: (30, 40, 50)
print(t[::2])  # Output: (10, 30, 50)
```

---

## Tuple Packing and Unpacking
Tuples support two handy features:

### Packing
Combine multiple values into a tuple without explicitly using parentheses:
```python
data = 1, "apple", 3.5
print(data)  # Output: (1, 'apple', 3.5)
```

### Unpacking
Split a tuple’s items into separate variables:
```python
x, y, z = data
print(x, y, z)  # Output: 1 apple 3.5
```

This is often used in functions to return multiple values:
```python
def get_range(numbers):
    return min(numbers), max(numbers)

result = get_range([10, 20, 30, 40])
print(result)  # Output: (10, 40)
```

---
## Tuple Immutability
Tuples are immutable, meaning their elements cannot be modified after creation.
```python
t = (10, 20, 30, 40)
t[1] = 70  # Raises TypeError: 'tuple' object does not support item assignment
```

---

## Mathematical Operations on Tuples
Tuples support operations like concatenation and repetition.

### 1. Concatenation (`+`)
Combine two tuples into one.
```python
t1 = (10, 20, 30)
t2 = (40, 50, 60)
t3 = t1 + t2
print(t3)  # Output: (10, 20, 30, 40, 50, 60)
```

### 2. Repetition (`*`)
Repeat the elements of a tuple.
```python
t1 = (10, 20, 30)
t2 = t1 * 3
print(t2)  # Output: (10, 20, 30, 10, 20, 30, 10, 20, 30)
```

### 3. Length of Tuple (`len()`)
Find the number of elements in a tuple.
```python
t = (10, 20, 30, 40)
print(len(t))  # Output: 4
```

### 4. Membership (`in`, `not in`)
Check if an item exists:
```python
t = (1, 2, 3)
print(2 in t)      # Output: True
print(5 not in t)  # Output: True
```

---

## Tuple Methods
Since tuples are immutable, they only have two main methods:
- **`count(value)`**: Counts how many times a value appears.
- **`index(value)`**: Returns the first position of a value. If the item is not found, it raises a `ValueError`.

```python
t = (1, 2, 2, 3)
print(t.count(2))  # Output: 2
print(t.index(3))  # Output: 3
```

You can explore all tuple methods using the `dir()` function:
```python
print(dir(tuple))
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

## Modifying Mutable Objects in a Tuple
While tuples themselves are immutable, they can contain mutable objects like lists, which *can* be modified:
```python
t = (1, 2, [3, 4])
t[2].append(5)
print(t)  # Output: (1, 2, [3, 4, 5])
```
The tuple still references the same list, but the list’s contents have changed.

---

## Conclusion
Tuples are a fast and efficient way to store ordered, unchanging collections of items in Python. Their immutability makes them perfect for fixed data, like constants or dictionary keys, and ensures data integrity. Features like packing and unpacking add flexibility, making tuples valuable in many scenarios, including machine learning tasks where fixed configurations or paired data are common. 

---

# 14. Set Data Structure in Python

A **set** is a mutable, unordered collection of unique elements, enclosed in curly braces `{}` or created with `set()`.

---

## What is a Set?

A **set** is an unordered collection data structure in Python that holds unique elements. It can store both homogeneous (same type) and heterogeneous (different types) elements but does not preserve the order of insertion.

## Creating a Set
1. A set can be created using:
  - Curly braces `{}` symbols
  - The `set()` predefined function
2. A set can store a group of objects or elements:
  - A set can store same (homogeneous) type of elements.
  - A set can store different (heterogeneous) types of elements.

## Key Characteristics of Sets
- **Unordered**: Elements in a set do not maintain a fixed order.
- **Unique Elements**: Duplicates are automatically removed.
- **Mutable**: You can modify a set after creation.
- **No Indexing**: Sets do not support indexing or slicing.
- **Dynamic Size**: The size of a set grows dynamically as elements are added.

Sets are **mutable**, meaning you can add or remove items after creating them, but they don’t support indexing (e.g., `set[0]` won’t work).

### Why Use Sets in ML/DL?
In ML and DL, sets are valuable for:
- Storing **unique labels** or classes in a dataset (e.g., `{ "cat", "dog" }`).
- Removing **duplicate samples** or features from data (e.g., cleaning a list of tokens in NLP).
- Efficiently checking if a feature or token exists (e.g., vocabulary in text processing).

---

## Creating a Set

You can create a set in two simple ways:

### 1. Using Curly Braces `{}`
List the items inside curly braces, separated by commas:
```python
s = {1, 2, 3, 4}
print(s)  # Output: {1, 2, 3, 4}
```

Duplicates are automatically removed:
```python
s = {1, 2, 2, 3}
print(s)  # Output: {1, 2, 3}
```

### 2. Using the `set()` Function
Convert any iterable (like a list or tuple) into a set:
```python
lst = [1, 2, 2, 3]
s = set(lst)
print(s)  # Output: {1, 2, 3}
```

### 3. Creating an Empty Set
Using `{}` creates an empty **dictionary**, not a set. Use `set()` instead:
```python
empty_set = set()
print(empty_set)  # Output: set()
```


---

## Key Features of Sets

| Feature            | Description                          |
|--------------------|--------------------------------------|
| **Syntax**         | `{}` or `set()`                     |
| **Duplicates**     | Not allowed—automatically removed   |
| **Order**          | Unordered (no fixed arrangement)    |
| **Mutability**     | Can add/remove items                |
| **Indexing**       | Not supported (no `s[0]`)           |

### Example in ML Context
Imagine you have a list of predicted labels with duplicates: `["cat", "dog", "cat", "bird"]`. Converting it to a set gives you unique labels:
```python
labels = ["cat", "dog", "cat", "bird"]
unique_labels = set(labels)
print(unique_labels)  # Output: {'cat', 'dog', 'bird'}
```

---

## Set Methods
- `set` is a predefined class containing multiple methods.
- You can view all available methods using the `dir(set)` function.

Since sets are mutable, you can modify them easily.
- `add(x)`: Add an element.
- `remove(x)`: Remove an element (raises KeyError if not found).
- `discard(x)`: Remove if present (no error).
- `union()`/`|` : Combine sets.
- `intersection()`/`&`: Common elements.
- `difference()`/`-`: Elements in one but not the other.

### Adding Items
Use the `add()` method to insert one item:
```python
s = {1, 2, 3}
s.add(4)
print(s)  # Output: {1, 2, 3, 4}
```

### Removing Items
- **`remove(item)`**: Deletes the item. Raises a `KeyError` if the item isn’t found.
- **`discard(item)`**: Deletes the item if it exists; no error if it doesn’t.
```python
s = {1, 2, 3}
s.remove(2)
print(s)  # Output: {1, 3}

s.discard(5)  # No error, even though 5 isn’t there
print(s)  # Output: {1, 3}
```

### Clearing All Items
Use `clear()` to empty the set:
```python
s.clear()
print(s)  # Output: set()
```

---

## Set Operations

Sets support mathematical operations, which are handy for comparing collections in ML/DL tasks like feature analysis.

### 1. Union (`|`)
Combines all items from two sets:
```python
a = {1, 2, 3}
b = {3, 4, 5}
print(a | b)  # Output: {1, 2, 3, 4, 5}
```
**ML Use**: Merge unique features from two datasets.

### 2. Intersection (`&`)
Finds items common to both sets:
```python
print(a & b)  # Output: {3}
```
**ML Use**: Identify shared vocabulary between two text corpora.

### 3. Difference (`-`)
Items in the first set but not the second:
```python
print(a - b)  # Output: {1, 2}
```
**ML Use**: Find features unique to one dataset.

### 4. Symmetric Difference (`^`)
Items in either set, but not both:
```python
print(a ^ b)  # Output: {1, 2, 4, 5}
```
**ML Use**: Highlight differences between two sets of labels.

---

## Checking Membership

Use `in` or `not in` to test if an item exists in a set. This is **fast** (O(1) time complexity) due to sets using hash tables internally:
```python
s = {1, 2, 3}
print(2 in s)      # Output: True
print(4 not in s)  # Output: True
```
**ML Use**: Quickly check if a token exists in a vocabulary set during text preprocessing.

---

## Set Comprehensions

Create sets concisely using set comprehensions:
```python
s = {x**2 for x in range(5)}
print(s)  # Output: {0, 1, 4, 9, 16}
```
**ML Use**: Generate a set of unique feature values (e.g., squared distances) for a model.

---

## Removing Duplicates from a List

A common ML task is cleaning data. Sets make it easy to remove duplicates:
```python
data = [1, 2, 2, 3, 3, 4]
unique_data = set(data)
print(unique_data)  # Output: {1, 2, 3, 4}
```
**Note**: Order isn’t preserved. If order matters, use a list or dictionary instead.

---

## Frozen Sets

A **frozenset** is an **immutable version** of a set. Once created, it can’t be changed:
```python
fs = frozenset([1, 2, 3])
print(fs)  # Output: frozenset({1, 2, 3})
```
**ML Use**: Use frozensets as dictionary keys when mapping fixed sets of features to values.

---

## When to Use Sets in ML/DL?

- **Data Cleaning**: Remove duplicate samples or features.
- **Feature Engineering**: Store unique categories or tokens.
- **Efficiency**: Perform fast membership checks or set operations (e.g., comparing datasets).

For example, in NLP, you might use a set to store a vocabulary:
```python
tokens = ["the", "cat", "the", "dog"]
vocab = set(tokens)
print(vocab)  # Output: {'the', 'cat', 'dog'}
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

---

## Conclusion

Sets are a simple yet powerful tool in Python, perfect for handling unique collections efficiently. Their ability to eliminate duplicates, perform fast membership tests, and support set operations makes them a go-to choice in ML and DL workflows. By mastering sets, you can streamline data preprocessing and improve your code’s readability and performance.

--- 

# 15. Dictionary Data Structure in Python

A **dictionary** is a mutable, unordered collection of key-value pairs, enclosed in curly braces `{}`.

A **dictionary** in Python is a collection of **key-value pairs**. It is a **mutable** and **unordered** data structure that allows efficient storage and retrieval of data based on unique keys.

**Key Features**
- Keys must be immutable (e.g., strings, numbers, tuples)
- Values can be any type
- Fast lookups via hashing

---

## Introduction

Each key is unique and linked to a specific value, like how a word in a real-world dictionary connects to its meaning.

In ML and DL, dictionaries are widely used for:
- Storing **model settings** (e.g., learning rate or batch size).
- Mapping **features** to their values in datasets.
- Holding **word-to-number mappings** in natural language processing (NLP).

---

## Creating a Dictionary
1. A dictionary can be created using:
  - **Curly braces `{}`**
  - **`dict()` predefined function**
2. **Key-Value Pairs:** Each item is a pair consisting of a key and its corresponding value.
3. **No Duplicate Keys:** Dictionary keys must be unique. Duplicate keys are not allowed, but duplicate values can be stored.
4. **Order:** Dictionaries are unordered, meaning the insertion order is not guaranteed (prior to Python 3.7).
5. **Mutable:** Dictionaries are mutable, meaning their contents can be changed after creation.
6. **No Indexing:** Elements are not stored in index order, and indexing or slicing doesn’t apply.

---

## Creating a Dictionary

You can create a dictionary in different ways depending on your needs.

### 1. Using Curly Braces `{}`
Use curly braces to list key-value pairs:
```python
# Example: Model settings in ML
settings = {"learning_rate": 0.01, "batch_size": 64}
print(settings)
# Output: {'learning_rate': 0.01, 'batch_size': 64}
```

### 2. Using `dict()`
The `dict()` function creates a dictionary from pairs or keyword arguments:
```python
# Using keyword arguments
model = dict(loss="cross_entropy", optimizer="sgd")
print(model)
# Output: {'loss': 'cross_entropy', 'optimizer': 'sgd'}

# From a list of pairs
labels = dict([(1, "positive"), (0, "negative")])
print(labels)
# Output: {1: 'positive', 0: 'negative'}
```

### 3. Starting Empty
Create an empty dictionary and add data later:
```python
data = {}
data["feature1"] = 10
data["feature2"] = 20
print(data)
# Output: {'feature1': 10, 'feature2': 20}
```

---

## Key Properties of a Dictionary

Here are the main features of a dictionary:
- **Key-Value Pairs**: Data is stored as pairs (e.g., "name": "Alice").
- **Unique Keys**: Keys must be unique; duplicate keys are not allowed.
- **Mutable**: You can change, add, or remove items after creation.
- **Unordered**: Items don’t have a fixed order (though Python 3.7+ keeps insertion order).
- **No Indexing**: Use keys to access values, not numbers like in lists.
- **Flexible Size**: Grows or shrinks as you add or remove items.

---

## When to Use a Dictionary?

Dictionaries shine when you need:
- **Fast Lookups**: Quickly find values using keys.
- **Key-Value Data**: Store related pairs, like settings or mappings.
- **ML/DL Examples**:
  - Mapping words to IDs in NLP (e.g., {"good": 1, "bad": 2}).
  - Storing feature values in a dataset (e.g., {"age": 30, "income": 50000}).

---

## Accessing Values in a Dictionary

Use a key to get its value. If the key doesn’t exist, Python raises an error unless you use a safer method.

### Basic Access
```python
settings = {"learning_rate": 0.01, "batch_size": 64}
print(settings["batch_size"])
# Output: 64
```

### Safe Access with `get()`
The `get()` method avoids errors by returning a default value if the key is missing:
```python
print(settings.get("optimizer", "adam"))
# Output: adam (default since "optimizer" isn’t in the dictionary)
```

### Looping Through a Dictionary
Use the `keys()`, `values()`, and `items()` methods:
You can access keys, values, or both:
```python
for key in settings:
    print(key, settings[key])
# Output: learning_rate 0.01
#         batch_size 64

# Using items() for key-value pairs
for key, value in settings.items():
    print(key, value)
```

---

## Updating a Dictionary

Since dictionaries are mutable, you can add new pairs or change existing ones.

### Adding a New Pair
```python
settings["epochs"] = 100
print(settings)
# Output: {'learning_rate': 0.01, 'batch_size': 64, 'epochs': 100}
```

### Changing a Value
```python
settings["learning_rate"] = 0.005
print(settings["learning_rate"])
# Output: 0.005
```
In ML, this is handy for tweaking model settings during experiments.

---

## Removing Elements from a Dictionary

You can delete specific pairs or clear the dictionary entirely.

### 1. Using `del` Keyword
Removes a specific key-value pair:
```python
del settings["batch_size"]
print(settings)
# Output: {'learning_rate': 0.005, 'epochs': 100}
```

### 2. Using `pop()`
Removes a key and returns its value:
```python
epochs = settings.pop("epochs")
print(epochs)  # Output: 100
print(settings)  # Output: {'learning_rate': 0.005}
```

### 3. Using `clear()`
Empties the dictionary:
```python
settings.clear()
print(settings)
# Output: {}
```

### 4. Deleting the Entire Dictionary
```python
# Deleting the entire dictionary object
del d
# Now, d is no longer accessible.
```

---

## Finding the Length of a Dictionary

The `len()` function counts the number of pairs:
```python
data = {"age": 25, "name": "Alice"}
print(len(data))
# Output: 2
```
In ML, this might show how many features are in a sample.

---

## Important Dictionary Methods

Dictionaries have useful methods for managing data. Here’s a table of key ones:

| Method       | Description                              | Example Use in ML/DL                  |
|--------------|------------------------------------------|---------------------------------------|
| `clear()`    | Removes all pairs                       | Reset a feature map                  |
| `get(key)`   | Gets a value, with optional default     | Safely fetch a model parameter       |
| `keys()`     | Returns all keys                        | List feature names in a dataset      |
| `values()`   | Returns all values                      | Extract feature values               |
| `items()`    | Returns all key-value pairs             | Loop through word-ID mappings in NLP |
| `pop(key)`   | Removes a key and returns its value     | Drop a feature and use its value     |
| `update()`   | Adds pairs from another dictionary      | Merge new settings into a model      |

### Example
```python
# Creating a dictionary
student = {"name": "Alice", "age": 20, "grades": [85, 90, 88]}

# Accessing and modifying
print(student["name"])  # Output: Alice
student["age"] = 21
print(student.get("id", "N/A"))  # Output: N/A

# Adding and removing
student["major"] = "CS"
del student["grades"]
print(student.pop("age"))  # Output: 21
print(student)  # Output: {'name': 'Alice', 'major': 'CS'}

# Iterating
for key, value in student.items():
    print(f"{key}: {value}")
# Output:
# name: Alice
# major: CS
```

---

## Dictionary Comprehension
Dictionary comprehension is a concise way to create a dictionary from an iterable object (list, set, tuple, etc.).

This is a short way to create dictionaries from other data, useful in ML for preprocessing.

### Example: Mapping Features to Indices
```python
features = ["age", "income", "score"]
feature_map = {f: i for i, f in enumerate(features)}
print(feature_map)
# Output: {'age': 0, 'income': 1, 'score': 2}
```
This is common in ML to assign numbers to feature names.

---

## Summary of Dictionary Features

| Feature            | Description                          |
|--------------------|--------------------------------------|
| **Syntax**         | `{key: value}` or `dict()`           |
| **Mutability**     | Can change after creation            |
| **Duplicates**     | Keys must be unique; values can repeat |
| **Order**          | Unordered (insertion order in 3.7+)  |
| **Indexing**       | Access by keys, not numbers          |
| **Use Case**       | Fast lookups with key-value pairs    |

---

## Applications in Machine Learning (ML) and Deep Learning (DL)
Dictionaries are widely used in ML/DL for:
- **Storing hyperparameters** (e.g., learning rate, batch size).
- **Mapping class labels** to their corresponding indices.
- **Representing word embeddings** in natural language processing (NLP).

### Example: Hyperparameters in ML
```python
hyperparams = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
}
print(hyperparams["learning_rate"])  # Output: 0.001
```

---
## Conclusion

Dictionaries are a powerful tool in Python for storing and managing key-value data. In ML and DL, they help with tasks like organizing model settings, mapping features, and handling data efficiently. Understanding dictionaries makes your code cleaner and faster.

--- 
## **Tools and Methods**
- **`len(obj)`**: Returns the length of a data structure.
  ```python
  print(len([1, 2, 3]))  # Output: 3
  ```
- **`in` Operator**: Checks membership.
  ```python
  print(2 in {1, 2, 3})  # Output: True
  ```
- **`sorted(iterable)`**: Returns a sorted list.
  ```python
  print(sorted({3, 1, 2}))  # Output: [1, 2, 3]
  ```
- **`copy()`**: Shallow copy for mutable structures.
  ```python
  lst = [1, [2, 3]]
  lst_copy = lst.copy()
  lst_copy[1][0] = 4
  print(lst)  # Output: [1, [4, 3]] (nested list is still linked)
  ```
- **`deepcopy()` from copy module`**: Deep copy for nested structures.
  ```python
  from copy import deepcopy
  lst_copy = deepcopy(lst)
  ```



