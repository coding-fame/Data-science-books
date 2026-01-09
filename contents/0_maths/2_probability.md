# Probability in ML

Probability helps us measure uncertainty and randomness, which are common in data.

---

## Introduction to Probability in ML

In ML and DL, we often work with incomplete or noisy data. Probability helps us:
- Handle uncertainty (e.g., predicting if an email is spam).
- Quantify randomness in data features, labels, and predictions.
- Build models that make smart decisions.

---

## Fundamental Concepts

### What is Probability?
Probability measures how likely an event is to happen. It ranges from:
- **0**: Impossible (e.g., rolling a 7 on a six-sided die).
- **1**: Certain (e.g., the sun rising tomorrow).

### Random Experiments and Sample Space
- A **random experiment** is an action with an uncertain result (e.g., flipping a coin (heads or tails?)).
- The **sample space** is all possible outcomes (e.g., {Heads, Tails} for a coin flip).

### Probability of an Event
The probability of an event \( A \) is:
```math\[
P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}
\]
```
- **Example**: Probability of rolling a 3 on a fair die is \( \frac{1}{6} \).

---

## Random Variables
A **random variable (RV)** is a function that assigns numerical values to outcomes of a random experiment.

### Types of Random Variables
1. **Discrete Random Variables**:
   - Countable values (e.g., number of heads in two coin flips: 0, 1, or 2).
2. **Continuous Random Variables**:
   - Any value in a range (e.g., height: 1.7m, 1.71m, etc.).

### Importance in ML
- RVs model data like:
  - Features (e.g., pixel values in an image).
  - Labels (e.g., spam or not spam).
  - Predictions (e.g., chance of rain).

---

## Expected Value (Mean)

The **expected value (E[X])** is the average outcome of a random variable over many trials.

### Calculation
For a discrete RV:
\[
E[X] = \sum (x \times P(X = x))
\]
- **Example**: Expected number of heads in two coin flips:
  - Outcomes: 0 heads ( \( \frac{1}{4} \) ), 1 head ( \( \frac{1}{2} \) ), 2 heads ( \( \frac{1}{4} \) ).
  - \( E[X] = (0 \times \frac{1}{4}) + (1 \times \frac{1}{2}) + (2 \times \frac{1}{4}) = 1 \).


### ML Application
- Used in cost functions to measure model performance (e.g., average prediction error).

---

## Independent and Dependent Events

- **Independent Events**: One event doesn’t change the other 
  - Example: Flipping a coin twice. The first flip (heads) does not affect the second flip.

- **Dependent Events**: One event affects the other.
  - Example: Drawing two cards from a deck without replacement. Drawing a red card first changes the probability of drawing another red card.

### ML Application
- Independence helps simplify models (e.g., Naive Bayes assumes features are independent).

---

## Rules of Probability

1. **Total Probability Rule**:
   - All probabilities in a sample space add to 1.
   - Example: For a die, \( P(1) + P(2) + \dots + P(6) = 1 \).

2. **Addition Rule (Mutually Exclusive Events)**:
   - \( P(A \text{ or } B) = P(A) + P(B) \) if \( A \) and \( B \) can’t happen together.
   - Example: Rolling a 1 or 2 on a die: \( \frac{1}{6} + \frac{1}{6} = \frac{1}{3} \).

3. **General Addition Rule**:
   - \( P(A \text{ or } B) = P(A) + P(B) - P(A \text{ and } B) \) for overlapping events.
   - Example: Drawing a king or a red card includes overlap (red kings).

4. **Complement Rule**:
   - \( P(\text{not } A) = 1 - P(A) \).
   - Example: If \( P(\text{rain}) = 0.3 \), then \( P(\text{no rain}) = 0.7 \).

---

## Conditional Probability

**Conditional probability** is the chance of event \( A \) happening given that event \( B \) has occurred:
\[
P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}
\]

### Example
- Deck of 52 cards: 26 red cards, 4 kings (2 red kings).
- Probability of drawing a king given the card is red:
  - \( P(\text{King} \mid \text{Red}) = \frac{2}{26} = \frac{1}{13} \).

### ML Application
- Used in classification (e.g., predicting a label based on features in Naive Bayes).

---

## Bayes' Theorem

**Bayes' Theorem** updates probabilities with new information:
\[
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
\]

### Example
- A test for a disease is 99% accurate.
- Only 1% of people have the disease (\( P(\text{Sick}) = 0.01 \)).

If you test **positive**, what’s the actual probability that you’re sick?  P(Test) 
- **Bayes' theorem** calculates this using prior probabilities and test accuracy. P(Test | Sick)


### ML Application
- Used in Bayesian models to update predictions as new data comes in.

---


## Probability Distributions

**Probability distributions** describe how probabilities are allocated across all possible outcomes. Think of it as a map that tells you how chances are distributed.

- **Why They Matter in ML/DL**:
  - Help predict outcomes (e.g., spam or not spam).
  - Model uncertainty in data (e.g., noisy sensor readings).
  - Reveal patterns in features (e.g., pixel values in images).

- **Two Main Types**:
  - **Discrete**: For countable outcomes (e.g., number of clicks).
  - **Continuous**: For infinite values in a range (e.g., temperature).

## Discrete Probability Distributions

Discrete distributions deal with outcomes you can count, like rolling a die or flipping a coin.

### What They Are
- A discrete random variable has a set number of possible values.
- Example: Rolling a fair six-sided die gives outcomes 1, 2, 3, 4, 5, or 6, each with a probability of \( 1/6 \).

### Probability Mass Function (PMF)
- The PMF tells you the probability of each specific outcome.
- **Formula**: \( P(X = x) \) is the chance that \( X \) equals \( x \).
- **Rules**:
  - Probabilities are between 0 and 1.
  - They sum to 1: \( \sum P(X = x) = 1 \).
- **Example**: For a die, \( P(X = 3) = 1/6 \).

### Discrete Distributions
- **Bernoulli Distribution**:
  - Models a single trial (e.g., spam or not spam).
  - Example: \( X = 1 \) (spam), \( X = 0 \) (not spam), \( P(X=1) = 0.3 \).
- **Binomial Distribution**:
  - Models multiple Bernoulli trials (e.g., number of spam emails in 10 emails).
  - Example: \( X \sim \text{Binomial}(n=10, p=0.3) \).

### Visualization
Here’s the PMF for a fair die in a table:

| Outcome | 1   | 2   | 3   | 4   | 5   | 6   |
|---------|-----|-----|-----|-----|-----|-----|
| Probability | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

A bar chart would show equal bars for each outcome.

## Continuous Probability Distributions

Continuous distributions handle outcomes with infinite possibilities within a range, like height or time.

### What They Are
- A continuous random variable can take any value in a range.
- Example: Height might be 1.7m, 1.71m, 1.712m, etc.

### Probability Density Function (PDF)
- The PDF shows the "density" of probability across a range.
- **Key Point**: The probability of an exact value (e.g., height = 1.7m) is 0. Instead, we look at areas under the curve.
- **Formula**: \( f(x) \) is the density at \( x \).
- **Rules**:
  - \( f(x) \geq 0 \) (no negative density).
  - Total area under the curve = 1: \( \int_{-\infty}^{\infty} f(x) \, dx = 1 \).

### Continuous Distributions
- **Normal (Gaussian) Distribution**:
  - Bell-shaped curve, used in many ML algorithms (e.g., Linear Regression).
  - Example: Errors in predictions are often assumed to be normally distributed.
- **Exponential Distribution**:
  - Models time between events (e.g., waiting time for a bus).
  - Used in survival analysis and reliability engineering.

### Visualization
The PDF of a normal distribution is a smooth bell curve. The area under it between two points (e.g., 1.6m to 1.8m) gives the probability of that range.

---

## Distribution Functions: PMF, PDF, and CDF

These functions help us work with probabilities in different ways.

### 1. Probability Mass Function (PMF)
- **For Discrete Variables**: Gives the probability of exact values.
- **Example**: For a die, \( P(X = 4) = \frac{1}{6} \).

### 2. Probability Density Function (PDF)
- **For Continuous Variables**: Shows probability density.
- **Example**: In a normal distribution, the PDF peaks at the mean.

### 3. Cumulative Distribution Function (CDF)
- **For Both Types**: Shows the probability that \( X \) is less than or equal to a value \( x \).
- For discrete variables: Sum of PMF up to \( x \).
- For continuous variables: Integral of PDF up to \( x \).
- **Formula**: \( F(x) = P(X \leq x) \).
  - Discrete: \( F(x) = \sum_{k \leq x} P(X = k) \).
  - Continuous: \( F(x) = \int_{-\infty}^{x} f(t) \, dt \).
- **Example**: For a die, \( F(3) = P(X \leq 3) = \frac{3}{6} = 0.5 \).
- **Shape**: Steps for discrete, smooth S-curve for continuous.

---

## **6. Key Probability Principles in ML**
- **Law of Large Numbers**: Sample mean converges to true probability as \(n\) increases.
- **Central Limit Theorem**: Sums of random variables approach normality—basis for many assumptions.
- **Maximum Likelihood Estimation (MLE)**: Optimize model parameters to maximize data likelihood.
- **Posterior Probability**: Update beliefs with data (Bayesian ML).

---

## Applications in ML and DL

Probability distributions power many ML and DL techniques.

### PMF in Action
- **Classification**: Predicts discrete labels (e.g., cat or dog).
- **Naive Bayes**: Uses PMF to estimate class probabilities from features.

### PDF in Action
- **Density Estimation**: Models continuous data (e.g., Kernel Density Estimation).
- **Generative Models**: Creates new data (e.g., Gaussian Mixture Models in image generation).

### CDF in Action
- **Anomaly Detection**: Spots outliers by checking how likely a value is.
- **Statistical Testing**: Assesses model performance (e.g., p-values).

---

## Practical Example in Python

Below is a Python example to visualize PDF and CDF for a normal distribution.

### Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
```

### Generate Data
```python
# Generate 1000 data points from a normal distribution (mean=0, std=1)
data = np.random.normal(loc=0, scale=1, size=1000)
```

### Plot PDF
```python
# Plot PDF of the normal distribution
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, loc=0, scale=1)
plt.plot(x, pdf, label='PDF')
plt.title('Probability Density Function (PDF)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
```
*This plots a bell curve showing the density.*

### Plot CDF
```python
# Plot CDF of the normal distribution
cdf = norm.cdf(x, loc=0, scale=1)
plt.plot(x, cdf, label='CDF')
plt.title('Cumulative Distribution Function (CDF)')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()
plt.show()
```
*This plots an S-curve showing cumulative probability.*

---

### What It Means
- The PDF shows where data is most dense (peaks at 0).
- The CDF shows the probability of values up to any point (reaches 1 as \( x \) grows).

*Note*: No images are embedded here, but the code generates plots in Jupyter Book.

## Key Takeaways

1. **Probability** quantifies uncertainty using numbers from 0 to 1.
2. **Random Variables** model outcomes in ML, either discrete or continuous.
3. **Conditional Probability** and **Bayes' Theorem** update predictions based on new information.
4. **Probability Distributions**: Map out how likely outcomes are.
5. **Discrete**: Use PMF for countable values (e.g., dice rolls).
6. **Continuous**: Use PDF for ranges (e.g., heights).
7. **CDF**: Gives cumulative probabilities for both types.
8. **Applications** in ML include classification, density estimation, and anomaly detection.

