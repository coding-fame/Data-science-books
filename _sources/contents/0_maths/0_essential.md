
# Essential Maths

---

## Number Types
| Type | Description | Example |
|------|-------------|---------|
| **Whole Numbers** | Non-negative integers | 0, 1, 2, 3... |
| **Natural Numbers** | Positive integers | 1, 2, 3... |
| **Integers** | Whole numbers + negatives | -2, -1, 0, 1 |
| **Real Numbers** | All rational/irrational numbers | 3.14, √2, -5.5 |

**Why it matters in ML**: Real numbers are used for model parameters (e.g., neural network weights), while integers often represent categories or counts.

### Order of Operations (PEMDAS)
PEMDAS ensures calculations follow the correct sequence: Parentheses, Exponents, Multiplication and Division (left to right), Addition and Subtraction (left to right).


```python
# Example calculation using PEMDAS
result = 2 * ((3 + 2)**2 / 5) - 4
print(result)  # Output: 6.0
```

**Why it matters in ML**: Correct order is critical in formulas like cost functions or gradient updates—mistakes can break a model.

---

## Variables & Functions

### Variables
Variables store values and are essential in ML for handling data and parameters.

```python
# Declaring and using variables
# x = int(input("Enter a number: "))  # Example input: 5
x = 5
print(3 * x)  # Output: 15

# Using Greek letters (common in ML)
beta = 1.75  # Represents a coefficient
theta = 30.0  # Represents an angle or parameter
```

**ML Context**: Variables like `theta` often stand for weights or biases in models, while `x` might be an input feature.

### Functions
Functions define relationships between inputs and outputs, forming the basis of ML models.

```python
def linear(x):
    return 2 * x + 1

# Generate values for plotting
x_values = range(4)
y_values = [linear(x) for x in x_values]
print("Function outputs:", y_values)  # Output: [1, 3, 5, 7]
```

**ML Context**: This simple linear function could represent a regression model predicting `y` from `x`.

---

## Visualizing Functions

Visualizations make mathematical relationships easier to understand, especially in ML.

### 2D Plot: Quadratic Function
A quadratic function like \( y = x^2 + 1 \) shows a curved relationship.

```python
from sympy import symbols, plot

x = symbols('x')
parabola = x**2 + 1
plot(parabola, title="Quadratic Function")
```

**Description**: This code generates a parabola, a U-shaped curve starting at \( y = 1 \) when \( x = 0 \).

**ML Context**: Quadratic shapes appear in loss functions like Mean Squared Error, where errors grow faster as predictions stray from targets.

### 3D Plot: Linear Plane
A function with two variables, like \( z = 2x + 3y \), forms a plane in 3D space.

```python
from sympy.plotting import plot3d

x, y = symbols('x y')
plane = 2*x + 3*y
plot3d(plane, title="3D Function Visualization")
```

**Description**: This creates a 3D plot of a slanted plane, showing how \( z \) changes with \( x \) and \( y \).

**ML Context**: Planes can represent decision boundaries in multi-dimensional ML models.

---

## Summation & Exponential Operations

### Summation
Summation adds up values, a common operation in ML for aggregating data or errors.

```python
# Sum of 2*i for i from 1 to 5
print(sum(2*i for i in range(1, 6)))  # Output: 30

# Sum over a list
values = [1, 4, 6, 2]
print(sum(10*x for x in values))  # Output: 130
```

**ML Context**: Summation is used in cost functions to combine errors across a dataset.

### Exponent Rules
Exponents describe rapid growth or decay, key in many ML algorithms.

| Rule | Example | Result |
|------|---------|--------|
| Product | x² * x³ | x⁵ |
| Quotient | x⁵ / x² | x³ |
| Power | (x²)³ | x⁶ |

```python
from sympy import simplify, symbols

x = symbols('x')
expr = x**2 / x**5
print(simplify(expr))  # Output: x^(-3)
```

**ML Context**: Exponents are used in activation functions (e.g., sigmoid) and learning rate decay.

---

## Logarithms & Exponential Functions

### Logarithms
Logarithms reverse exponents and help manage large value ranges in ML.

```python
from math import log

# Log base 2 of 8
print(log(8, 2))  # Output: 3.0 (since 2^3 = 8)

# Natural log (base e)
print(log(10))  # Output: 2.302585092994046
```

**ML Context**: Logarithms appear in loss functions like cross-entropy, measuring prediction accuracy.

### Exponential Functions
Exponential functions model rapid changes, such as growth or decay.

```python
from math import exp

# Discrete compound interest
principal = 100
rate = 0.20
print(principal * (1 + rate/365)**(365*2))  # Output: ≈149.175

# Continuous compound interest
print(principal * exp(rate*2))  # Output: ≈149.182
```

**ML Context**: Exponential decay adjusts learning rates in training, helping models converge smoothly.

---

## Calculus Fundamentals

### Limits
Limits describe how functions behave as inputs approach specific values.

```python
from sympy import limit, symbols, oo

x = symbols('x')
print(limit(1/x, x, oo))  # Output: 0
```

**ML Context**: Limits help analyze convergence in optimization algorithms like gradient descent.

### Derivatives
Derivatives measure how fast a function changes, critical for ML optimization.

```python
from sympy import diff

x = symbols('x')
f = x**2
print(diff(f, x))  # Output: 2x
print(diff(f, x).subs(x, 2))  # Output: 4
```

**ML Context**: Derivatives guide weight updates in neural networks via backpropagation.

### Partial Derivatives
Partial derivatives handle functions with multiple variables.

```python
from sympy.plotting import plot3d

x, y = symbols('x y')
surface = 2*x**3 + 3*y**3
plot3d(surface, title="Partial Derivative Visualization")
```

**Description**: This plots a 3D surface, showing how \( z \) depends on \( x \) and \( y \).

**ML Context**: Partial derivatives compute gradients for multi-parameter models.

---

## ∫ Integration Techniques

### Numerical Integration
Integration finds areas under curves, useful in probability and ML.

```python
def integrate(f, a, b, n=1000000):
    delta = (b - a) / n
    return sum(f(a + i*delta) * delta for i in range(n))

print(integrate(lambda x: x**2 + 1, 0, 1))  # Output: ≈1.333333
```

**ML Context**: Numerical integration approximates probabilities in continuous distributions.

### Symbolic Integration
Symbolic integration gives exact solutions for integrals.

```python
from sympy import integrate, symbols

x = symbols('x')
result = integrate(x**2 + 1, (x, 0, 1))
print(result)  # Output: 4/3 (≈1.333333)
```

**ML Context**: Integration is used in Bayesian methods to compute expected values.

---

## Key Formula Cheat Sheet

| Concept           | Formula                          | Python Equivalent         |
|-------------------|----------------------------------|---------------------------|
| Compound Interest | \( A = P(1 + \frac{r}{n})^{nt} \) | `p * (1 + r/n)**(n*t)`   |
| Natural Log       | \( \ln(x) \)                    | `math.log(x)`            |
| Derivative        | \( \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \) | `sympy.diff(f, x)` |
| Chain Rule        | \( \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} \) | `diff(z.subs(y, f(x)), x)` |
| Summation         | \( \sum_{i=1}^{n} a_i \)        | `sum(a)`                 |
| Integration       | \( \int_a^b f(x) \, dx \)       | `integrate(f, (x, a, b))` |

**Tip**: Use SymPy for symbolic math and Matplotlib for plots to boost your ML workflow!


