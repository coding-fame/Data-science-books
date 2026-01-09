

# Regular Expressions

This guide explains **Regular Expressions (Regex)** in Python, a powerful tool for working with text. 

---

## What Are Regular Expressions?

A **regular expression** is a sequence of characters that defines a search pattern. It helps you find, match, or validate strings based on specific rules.

### Why Use Regex in ML and DL?
Regex is valuable in ML and DL for:
- **Cleaning Data**: Removing unwanted text, like special characters or extra spaces.
- **Extracting Features**: Finding patterns like emails or phone numbers in text.
- **Text Preprocessing**: Preparing data for Natural Language Processing (NLP) tasks.

For example, regex can extract dates from customer reviews to analyze trends in an ML model.

---

## When to Use Regular Expressions

Use regex when you need to:
- Match strings with a specific format (e.g., 10-digit phone numbers).
- Check if inputs are valid (e.g., email addresses).
- Search for patterns in text (e.g., URLs or keywords).

### Examples:
- **Phone Numbers**: `[7-9]\d{9}` matches 10-digit numbers starting with 7, 8, or 9.
- **Emails**: `\w+@\w+\.\w+` matches simple email formats like `user@domain.com`.

---

## Key Uses of Regular Expressions

Regex is helpful in many areas:
1. **Validation**: Ensuring data follows rules (e.g., strong passwords).
2. **Pattern Matching**: Finding text patterns (e.g., search tools like `grep`).
3. **Text Processing**: Preparing text for ML models.
4. **Compilers**: Analyzing code syntax.
5. **Protocols**: Validating messages in networks like TCP/IP.

In ML and DL, regex is key for tasks like tokenizing text or validating datasets.

---

## Python’s `re` Module

Python’s `re` module makes regex easy to use. Here are some important functions:

- **`compile()`**: Turns a regex pattern into an object for reuse.
  ```python
  import re
  pattern = re.compile("ab")
  ```

- **`finditer()`**: Finds all matches and returns them as an iterator.
  ```python
  matcher = pattern.finditer("abaababa")
  ```

- **Match Object Methods**:
  - `start()`: Where the match begins.
  - `end()`: Where the match ends (plus 1).
  - `group()`: The matched text.

### Example: Counting Matches
```python
import re
pattern = re.compile("ab")
matcher = pattern.finditer("abaababa")
count = 0
for match in matcher:
    count += 1
    print(f"Match at {match.start()} to {match.end()}: {match.group()}")
print(f"Total matches: {count}")
```

---

## Character Classes

Character classes let you match specific groups of characters:

- `[abc]`: Matches `a`, `b`, or `c`.
- `[^abc]`: Matches anything except `a`, `b`, or `c`.
- `[a-z]`: Matches any lowercase letter.
- `[0-9]`: Matches any digit.
- `[a-zA-Z0-9]`: Matches any alphanumeric character.

### Example: Finding Digits
```python
import re
matcher = re.finditer("[0-9]", "a7b@k9z")
for match in matcher:
    print(match.group())  # Output: 7, 9
```

In ML, this can extract numbers from text for feature engineering.

---

## Predefined Character Classes

These shortcuts make regex simpler:

- `\d`: Any digit (0-9).
- `\D`: Any non-digit.
- `\w`: Any word character (letters, digits, underscore).
- `\W`: Any non-word character (e.g., `@`, `#`).
- `\s`: Space character.
- `.`: Any character.

### Example: Finding Spaces
```python
import re
matcher = re.finditer(r"\s", "a7b k@9z")
for match in matcher:
    print(match.group())  # Output: space
```

In NLP, `\s` helps split text into words.

---

## Quantifiers

Quantifiers control how many times a pattern repeats:

- `a+`: One or more `a`s.
- `a*`: Zero or more `a`s.
- `a?`: Zero or one `a`.
- `a{3}`: Exactly 3 `a`s.
- `a{2,4}`: Between 2 and 4 `a`s.

### Example: Repeated Characters
```python
import re
matcher = re.finditer("a+", "abaabaaab")
for match in matcher:
    print(match.group())  # Output: a, aa, aaa
```

In DL, this can detect repeated patterns in text classification.

---

## Anchors: `^` and `$`

Anchors check string positions:
- `^x`: String starts with `x`.
- `x$`: String ends with `x`.

### Example: Checking Start
```python
import re
s = "Learning Python is easy"
if re.search("^Learn", s):
    print("Starts with 'Learn'")
```

In ML, anchors ensure data follows expected formats.

---

## Main `re` Module Functions

### 1. `match()`
Checks if a pattern matches at the string’s start.
```python
if re.match("ab", "abc"):
    print("Match at start")
```

### 2. `fullmatch()`
Checks if the entire string matches the pattern.
```python
if re.fullmatch("abc", "abc"):
    print("Full match")
```

### 3. `search()`
Finds the first match anywhere in the string.
```python
m = re.search("a", "banana")
if m:
    print(f"Found 'a' at {m.start()}")
```

### 4. `findall()`
Returns all matches as a list.
```python
print(re.findall("[0-9]", "a7b9c5"))  # Output: ['7', '9', '5']
```

### 5. `sub()`
Replaces matches with new text.
```python
s = re.sub("[a-z]", "#", "a7b9c5")
print(s)  # Output: #7#9#5
```

### 6. `split()`
Splits text based on a pattern.
```python
l = re.split(",", "apple,banana,cherry")
print(l)  # Output: ['apple', 'banana', 'cherry']
```

In ML, `split()` helps tokenize text datasets.

---

## Practical Examples for ML/DL

### 1. Validating Emails
```python
import re
email = input("Enter email: ")
if re.fullmatch(r"\w+@\w+\.\w+", email):
    print("Valid email")
else:
    print("Invalid email")
```
- Ensures clean email data for ML models.

### 2. Extracting Phone Numbers
```python
import re
with open("input.txt", "r") as f1, open("output.txt", "w") as f2:
    for line in f1:
        numbers = re.findall(r"[7-9]\d{9}", line)
        for n in numbers:
            f2.write(n + "\n")
print("Phone numbers saved to output.txt")
```
- Extracts contact info from unstructured text.

### 3. Checking Custom Formats
```python
import re
s = input("Enter string: ")
if re.fullmatch(r"[a-k][0369][a-zA-Z0-9#]*", s):
    print("Valid format")
else:
    print("Invalid format")
```
- Validates identifiers or codes for data processing.

---

## Conclusion

Regular expressions in Python are essential for handling text in ML and DL. They simplify:
- Data validation and cleaning.
- Feature extraction from text.
- Preparing inputs for models.


