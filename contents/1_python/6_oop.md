# Object-Oriented Programming

---

## Introduction to OOP

OOP is a programming approach that models real-world entities as **objects** with attributes (data) and behaviors (methods). Python supports OOP through:
- **Classes**: Blueprints for creating objects.
- **Objects**: Instances of classes.
- Core principles: **Encapsulation**, **Inheritance**, **Polymorphism**, and **Abstraction**.

### Why Use OOP?
OOP makes code:
- Easier to understand and manage.
- Reusable for different projects.
- Useful for modeling real-world things, like a car or a person.

In **Machine Learning (ML)** and **Deep Learning (DL)**, OOP is great for:
- Building models (e.g., a `NeuralNetwork` class).
- Reusing code (e.g., extending a base `Model` class).
- Hiding complex details (e.g., keeping model data private).

---

## Core OOP Concepts

Here’s what OOP is built on:

- **Class**: A plan for creating objects.
- **Object**: A real instance of a class with its own data.
- **Constructor**: A method to set up an object when it’s created.
- **Inheritance**: Making new classes based on old ones to reuse code.
- **Polymorphism**: Using the same method name for different tasks.
- **Encapsulation**: Keeping data safe by controlling access.
- **Abstraction**: Showing only what’s needed and hiding the rest.

---

## Classes

### **Definition**
1. A class is a **blueprint (model)** for creating objects; it does not exist physically.
2. A class is a **specification** (idea/plan/theory) of **Properties (Attributes/Data)** and **Actions (Methods/Behaviors)** of objects.

### What is a Class?
A **class** is a blueprint. It says what data (attributes) and actions (methods) an object will have, but it doesn’t hold data itself until you make an object.

#### Syntax
```python
class ClassName:
    """What this class does"""
    def __init__(self, param):
        self.attribute = param  # Data
    
    def method(self):
        print("Doing something!")
```

- **`class` keyword**: Starts the class.
- **`__init__`**: Sets up the object (called a constructor).
- **Methods**: Actions the object can do.

### **Key Characteristics**
- A class is created using the **`class` keyword**.
- A class contains three main components:
  - **Constructor**: Used for initialization.
  - **Properties**: Represent data (instance variables).
  - **Methods**: Represent actions or behavior.
- Class names follow the **PascalCase** naming convention.

#### Example
```python
class Dog:
    """A simple dog class"""
    def __init__(self, name):
        self.name = name
    
    def bark(self):
        print(f"{self.name} says Woof!")
```

### Naming Classes
Use **PascalCase** (e.g., `Dog`, `DataProcessor`) for class names.

---

## Objects

### **Definition**
1. An **object** is an instance of a class.
2. It stores multiple values and holds data for the class's attributes.
3. It represents a **real-world entity** (e.g., car, phone, user).

### What is an Object?
An **object** is an instance of a class. It’s like taking the blueprint and building something real with its own values.

#### Why Use Objects?
- **Hold data**: Each object has its own values.
- **Do actions**: Use methods to work with the data.
- **Access stuff**: Get or change attributes.

#### Syntax
```python
object_name = ClassName(params)
```

#### Example
```python
my_dog = Dog("Buddy")
my_dog.bark()  # Output: Buddy says Woof!
```

In ML, an object could be a model like `my_model = NeuralNetwork()`.

### **Key Notes**
- Objects **store values** for instance variables.
- Multiple objects = multiple constructor executions.
- Instance of a class **must be accessed** using objects.

---

## Constructors

### What is a Constructor?
A **constructor** is a special method called `__init__`. It runs automatically when you create an object and sets up its starting data.
- A **constructor** is a special method in Python used for initializing instance variables when an object is created.

### **Purpose**
- Initializes instance variables.
- Automatically executes on object creation.

### Types
1. **Default Constructor**: No extra inputs (except `self`).
  - Used when no additional data is needed for initialization.

   ```python
   class Box:
       def __init__(self):
           self.size = 10
   ```
2. **Parameterized Constructor**: Takes inputs to customize the object.
  - Allows passing parameters during object creation to initialize instance variables.
   ```python
    class ClassName:
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2
    ```

#### Example in ML
```python
class Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
```

---

## Instance Variables

### What are Instance Variables?
These are pieces of data tied to an object. Each object can have its own unique values for these variables.
- Instance variables are attributes that belong to an instance (object) of a class.
- Instance variables **belong to an object** of a class.
- Their values **can change** from one object to another.
- Declared using the **`self` keyword**.
- Accessed via **object reference**.

#### Example
```python
class Cat:
    def __init__(self, name, color):
        self.name = name    # Instance variable
        self.color = color  # Instance variable
```

- `self.name` and `self.color` change for each `Cat` object.

In DL, instance variables might store things like a layer’s weights.

---

## Instance Methods

### What are Instance Methods?
These are functions inside a class that use the object’s data. They always start with `self` to connect to the object.
- Instance methods **act upon instance variables** of a class.
- The **first parameter is always `self`**.
- These methods are bound to an object and can access instance variables.

#### Example
```python
class Square:
    def __init__(self, side):
        self.side = side
    
    def area(self):
        return self.side * self.side
```

- `area()` uses `self.side` to calculate the square’s area.

In ML, methods might do tasks like `train()` or `predict()`.

---

### **Instance Variables vs Methods**

| Feature            | Instance Variables          | Instance Methods          |
|--------------------|----------------------------|---------------------------|
| **Purpose**        | Store object state         | Define object behavior    |
| **Declaration**    | Using `self.attribute`     | `def method_name(self)`   |
| **Access**         | `object.attribute`         | `object.method()`         |
| **Memory**         | Unique per object          | Shared across instances   |

---

## Understanding `self`

### What is `self`?
`self` is how Python refers to the current object. It lets you work with the object’s data and methods inside the class.
- **`self` is a predefined variable** that refers to the current class instance (object).
- It is used to initialize instance variables and create instance methods.

#### Why Use `self`?
- **Set data**: `self.name = "Max"`
- **Use data**: `print(self.name)`
- **Call methods**: `self.bark()`
- **Creating constructors** (`__init__(self)`).
- **Defining instance methods** (`def method(self)`).
- Enables attribute/method access.

#### Example
```python
class Bird:
    def __init__(self, species):
        self.species = species
    
    def fly(self):
        print(f"The {self.species} is flying!")
```

- `self.species` links the data to the object.

---

## Methods vs. Constructors

| **Feature**         | **Constructor**         | **Method**             |
|---------------------|-------------------------|------------------------|
| **Name**            | Always `__init__`       | Any name you choose    |
| **Job**             | Sets up the object      | Does a task            |
| **When It Runs**    | When object is made     | When you call it       |
| **Returns**         | Nothing (None)          | Can return something   |

---

## Best Practices

- **Keep Data Safe**: Use `_` for private attributes (e.g., `_score`).
- **One Task per Class**: Each class should focus on one job.
- **Add Hints**: Use type hints like `def __init__(self, name: str)`.
- **Make It Readable**: Add `__str__` to show object details.
  ```python
  class Book:
      def __init__(self, title):
          self.title = title
      def __str__(self):
          return f"Book: {self.title}"
  ```

---

## OOP in ML/DL: Example

Here’s a simple `Model` class for ML:

```python
class Model:
    def __init__(self, name):
        self.name = name
    
    def train(self, data):
        print(f"Training {self.name} with {len(data)} samples.")
    
    def predict(self, input_data):
        return [1] * len(input_data)  # Simple prediction
```

- **Attributes**: `name`
- **Methods**: `train()`, `predict()`

This setup is similar to libraries like PyTorch or scikit-learn.

---

## Interview-Ready Concepts

### Common Questions
1. **Q:** What’s a class vs. an object?  
   **A:** A class is the plan; an object is the real thing made from it.

2. **Q:** Can you have multiple constructors?  
   **A:** Not really, but you can use default values:
   ```python
   def __init__(self, name="Unknown"):
   ```

3. **Q:** Why use OOP in ML?  
   **A:** It keeps code organized, reusable, and easy to update.

---

## Conclusion

- **Classes** create the structure; **objects** fill it with data.
- **Constructors** start the object; **methods** make it work.
- **`self`** ties everything to the object.
- OOP is powerful for clean, reusable code, especially in ML and DL.

--- 

# **Core OOP Principles**

## **a. Encapsulation**
Encapsulation restricts direct access to an object’s data, exposing only what’s necessary through methods. Python uses naming conventions for this:
- `_variable`: Protected (convention, not enforced).
- `__variable`: Private (name mangling).

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited {amount}. New balance: {self.__balance}"
        return "Invalid amount"

    def get_balance(self):  # Public method to access private data
        return self.__balance

account = BankAccount("Alice", 1000)
print(account.deposit(500))       # Output: Deposited 500. New balance: 1500
print(account.get_balance())      # Output: 1500
# print(account.__balance)        # Error: AttributeError
print(account._BankAccount__balance)  # Name mangling workaround: 1500
```

---

## **b. Inheritance**
Inheritance allows a class to inherit attributes and methods from another class.
- **Parent/Base Class**: The class being inherited from.
- **Child/Derived Class**: The class that inherits.

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "I make a sound!"

class Cat(Animal):  # Inherits from Animal
    def speak(self):  # Method overriding
        return f"{self.name} says Meow!"

class Dog(Animal):
    pass

cat = Cat("Whiskers")
dog = Dog("Buddy")
print(cat.speak())  # Output: Whiskers says Meow!
print(dog.speak())  # Output: I make a sound!
```

---
## **c. Polymorphism**
Polymorphism allows different classes to be treated as instances of the same parent class, often through method overriding.

```python
def make_animal_speak(animal):
    print(animal.speak())

animals = [Cat("Kitty"), Dog("Rex")]
for animal in animals:
    make_animal_speak(animal)
# Output:
# Kitty says Meow!
# I make a sound!
```

---

## **d. Abstraction**
Abstraction hides complex implementation details and exposes only the essentials. Python uses abstract base classes (ABCs) from the `abc` module.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

rect = Rectangle(4, 5)
print(rect.area())  # Output: 20
# shape = Shape()   # Error: Can't instantiate abstract class
```

---

# **Special Methods (Magic/Dunder Methods)**
Special methods customize object behavior and are denoted by double underscores (e.g., `__init__`, `__str__`).

## **Example**
```python
class Book:
    def __init__(self, title, pages):
        self.title = title
        self.pages = pages

    def __str__(self):  # String representation
        return f"{self.title} ({self.pages} pages)"

    def __len__(self):  # Length of object
        return self.pages

    def __add__(self, other):  # Adding two books
        return self.pages + other.pages

book1 = Book("Python 101", 200)
book2 = Book("OOP Basics", 150)
print(book1)          # Output: Python 101 (200 pages)
print(len(book1))     # Output: 200
print(book1 + book2)  # Output: 350
```

Common dunder methods:
- `__eq__`: Equality (`==`)
- `__lt__`: Less than (`<`)
- `__call__`: Makes an object callable

---
# **Properties (Getters, Setters, Deleters)**
The `@property` decorator provides a clean way to manage attribute access.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Radius cannot be negative")

    @radius.deleter
    def radius(self):
        del self._radius

circle = Circle(5)
print(circle.radius)  # Output: 5
circle.radius = 10    # Setter
print(circle.radius)  # Output: 10
# circle.radius = -1   # Error: ValueError
del circle.radius     # Deleter
```

---
# **6. Multiple Inheritance and Method Resolution Order (MRO)**
Python supports multiple inheritance, where a class can inherit from multiple parents. The MRO determines the order of method lookup.

```python
class A:
    def greet(self):
        return "Hello from A"

class B:
    def greet(self):
        return "Hello from B"

class C(A, B):  # Inherits from A and B
    pass

c = C()
print(c.greet())      # Output: Hello from A (A comes first in MRO)
print(C.__mro__)      # Output: (<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class 'object'>)
```

---

# **Dataclasses (Python 3.7+)**
Simplify class creation with the `dataclass` decorator.

```python
from dataclasses import dataclass

@dataclass
class Student:
    name: str
    id: int
    grade: float = 0.0

student = Student("Bob", 1)
print(student)  # Output: Student(name='Bob', id=1, grade=0.0)
```

---