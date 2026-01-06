
# Naïve Bayes Classifier

The Naïve Bayes Classifier is a popular *probabilistic*  supervised machine learning algorithm used for classification tasks. 
It is based on **Bayes' Theorem**. 

## **What is the Naïve Bayes Classifier?**
Naïve Bayes is a family of simple, probabilistic classifiers based on applying Bayes’ theorem with a “naïve” assumption of *conditional independence* between features given the class label. 
It predicts the class of a data point by calculating the posterior probability of each class and selecting the one with the highest probability.

---

## **Why Use Naïve Bayes?**
- **Simplicity**: Easy to implement and computationally efficient.
- **Probabilistic**: Provides probability estimates, not just predictions.
- **Scalability**: Performs well with high-dimensional data (e.g., text).
- **Robustness**: Works surprisingly well even when independence assumption is violated.


## **How It Works**
1. **Bayes’ Theorem**: `\( P(y|X) = P(X|y).P(y) / P(X) \)`.
   - \(P(y|X)\): Posterior probability of class \(y\) given features \(X\).
   - \(P(X|y)\): Likelihood of features given class.
   - \(P(y)\): Prior probability of class.
   - \(P(X)\): Evidence (often ignored during classification since as it’s constant across classes). 

2. **Naïve Assumption**: The "naïve" assumption is that all features \( x_1, x_2, ...... , x_n \) in \( X \) are independent given \(y\):
   ```math
    P(X|y) = P(x_1|y).P(x_2|y) ........ P(x_n|y)
   ```
- This assumption reduces computational complexity, making Naïve Bayes efficient even for large datasets.

3. **Prediction**:
```math
\hat{y} = \arg\max_y P(y) \cdot \prod_{i=1}^n P(x_i|y)
```

## **Key Concepts and Variants**

### **a. Variants of Naïve Bayes**
1. **Gaussian Naïve Bayes**:
   - Assumes continuous features follow a Gaussian (normal) distribution.
   - Commonly used for datasets with numeric features, like measurements.
   - Likelihood: 
    ```math
    P(x_i|C) = \frac{1}{\sqrt{2\pi\sigma_C^2}} \exp\left(-\frac{(x_i - \mu_C)^2}{2\sigma_C^2}\right)
    ```
where \( \mu_C \) and \( \sigma_C \) are the mean and variance of feature \( x_i \) for class \( C \).

2. **Multinomial Naïve Bayes**:
   - Suitable for **discrete features**, such as word counts in text data.
   - Often used in text classification tasks (e.g., spam detection).
   - Models the frequency or occurrence of events.
3. **Bernoulli Naïve Bayes**:
   - Designed for **binary/boolean features** (e.g., 0 or 1 indicating absence or presence).
   - Useful in text classification when only the presence of a feature matters, not its frequency.

### **b. Hyperparameters**
- **Smoothing**: Prevents zero probabilities (e.g., Laplace smoothing with `alpha` in MultinomialNB).
- **Prior**: Can specify class priors (`prior` in `scikit-learn`), otherwise estimated from data.

---

## Advantages and Limitations

### Advantages
- **Efficiency**: Fast training and prediction, even with large datasets.
- **Simplicity**: Easy to implement and interpret.
- **Scalability**: Handles high-dimensional data well (e.g., text with many words).

### Limitations
- **Independence Assumption**: May fail if features are strongly correlated.
- **Limited Expressiveness**: Less powerful than complex models (e.g., neural networks) for some tasks.

---

## **Practical Examples**

### **Example 1: Classification with Gaussian Naïve Bayes (Iris Dataset)**

The Iris dataset contains measurements of flowers (sepal length, sepal width, petal length, petal width) labeled into three species.

```python
# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Class labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Gaussian Naïve Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Output**: You’ll see an accuracy (e.g., 0.96) and a detailed report with precision, recall, and F1-score for each class.

---

### **Example 2: Text Classification with Multinomial Naïve Bayes**
Let’s classify short movie reviews as **positive (1)** or **negative (0)** based on word counts.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

# Synthetic text data
texts = ["good movie", "bad film", "great show", "terrible plot"]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Fit Multinomial NB
mnb = MultinomialNB(alpha=1.0)  # Laplace smoothing
mnb.fit(X, labels)

# Predict
test_text = ["good show"]
X_test = vectorizer.transform(test_text).toarray()
print("Prediction:", mnb.predict(X_test))  # Output: [1]
print("Probabilities:\n", mnb.predict_proba(X_test))

# Evaluation
y_pred = mnb.predict(X)
print("Classification Report:\n", classification_report(labels, y_pred))
```

**Output**: Likely `1` (positive), as "good show" aligns with positive sentiment.

---

### **Example 3: Binary Features with Bernoulli Naïve Bayes**
```python
from sklearn.naive_bayes import BernoulliNB

# Synthetic binary data
X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]])
y = np.array([0, 1, 1, 0])

# Fit Bernoulli NB
bnb = BernoulliNB()
bnb.fit(X, y)

# Predict
print("Predictions:", bnb.predict(X))  # Output: [0 1 1 0]
print("Feature Log Probabilities:\n", bnb.feature_log_prob_)
```

---

## **Tools and Methods Summary**
- **Modeling**: `sklearn.naive_bayes.GaussianNB`, `MultinomialNB`, `BernoulliNB`.
    - `GaussianNB`: For continuous features.
    - `MultinomialNB`: For discrete counts (e.g., text).
    - `BernoulliNB`: For binary features.
- **Text Processing**: `sklearn.feature_extraction.text.CountVectorizer`.
    - `CountVectorizer`: Converts text into word count matrices.
    - `TfidfVectorizer`: Alternative for weighting words by importance (can be used with MultinomialNB).
- **Evaluation**: `sklearn.metrics.accuracy_score`, `classification_report`.
    - `accuracy_score`: Overall correctness.
    - `classification_report`: Detailed precision, recall, and F1-score.
- **Visualization**: `matplotlib.pyplot.contourf()`, `seaborn.scatterplot()`.
- **Handling Zero Probabilities**:
   - Use **Laplace smoothing** (enabled by default in scikit-learn with `alpha=1.0`) to avoid zero likelihoods when a feature value is absent in training data.

```python
from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```

---

## How Naïve Bayes Works: A Hands-On Example

Let’s explore how Naïve Bayes classifies data with a simple example: classifying emails as **spam** or **not spam** based on the presence of two words, "free" and "offer."

### Training Data
We have six emails with binary features (1 = word present, 0 = word absent):
- **Spam**: [1, 1], [1, 0], [0, 1] (3 emails)
- **Not Spam**: [0, 0], [0, 1], [1, 0] (3 emails)

### **The Goal**:  
We want to guess if a new email (with both "free" and "offer") is spam or not spam, using past data.

---

### **Step 1: Basic Chances**  
- Half of the emails in our training data are spam, and half are not.  
  → **Chance of spam** = 50%  
  → **Chance of not spam** = 50%  

---

### **Step 2: Learn Word Patterns**  
We check how often "free" and "offer" appear in spam vs. not-spam emails:  

- **In spam emails**:  
  - "free" appears in **2 out of 3** spam emails.  
  - "offer" appears in **2 out of 3** spam emails.  

- **In not-spam emails**:  
  - "free" appears in **1 out of 3** not-spam emails.  
  - "offer" appears in **1 out of 3** not-spam emails.  

---

### **Step 3: Guess the New Email**  
The new email has **both "free" and "offer"**. We calculate two scores:  

1. **Spam Score**:  
   = (Chance of "free" in spam) × (Chance of "offer" in spam) × (Chance of spam)  
   = \( \frac{2}{3} \times \frac{2}{3} \times 0.5 = 0.222 \)  

2. **Not-Spam Score**:  
   = (Chance of "free" in not-spam) × (Chance of "offer" in not-spam) × (Chance of not-spam)  
   = \( \frac{1}{3} \times \frac{1}{3} \times 0.5 = 0.056 \)  

---

### **Result**:  
The spam score (**0.222**) is higher than the not-spam score (**0.056**), so the email is marked as **spam**.  

---

### **Why It’s Called "Naïve"**:  
It assumes "free" and "offer" don’t affect each other (even though in real life, they might). This simplification helps make quick guesses, even if it’s not 100% realistic.

---