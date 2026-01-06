# Descriptive Statistics

---

# 1. Measures of Central Tendency

Measures of central tendency are single values that represent the central point of a dataset, providing a summary of its typical or average behavior. They answer the question: *“What is the most representative value in this data?”*

**Purpose**: Identify the "center" or typical value of a dataset to inform data preprocessing and model decisions in machine learning (ML) and deep learning (DL).

---

## 1. Mean (The "Average")

- **Definition**: The sum of all values divided by the number of values.
- **Formula**: `Mean = Σx_i / n`, where `x_i` is each value and `n` is the total count.
- **Example**:
  - Test scores: 80, 90, 100
  - Mean = (80 + 90 + 100) ÷ 3 = 90
- **Best Use Case**: When all values are equally important, such as:
  - Average temperature
  - Average student grades
- **ML/DL Application**:
  - Used in loss functions like **Mean Squared Error (MSE)**, which measures how close a model's predictions are to the actual values by averaging the squared differences.
  - Example: In regression models, MSE helps evaluate model accuracy.

---

## 2 Median (The "Middle")

- **Definition**: The middle value when data is sorted in ascending order.
  - For an odd number of values, it's the middle value.
  - For an even number of values, average the two middle values.
- **Example**:
  - House prices: $100k, $200k, $300k → Median = $200k
  - For an even count:
    - $100k, $200k, $300k, $400k → Median = ($200k + $300k) ÷ 2 = $250k
- **Best Use Case**: When data has outliers, such as:
  - Income (e.g., a billionaire skews the mean)
  - Home prices (e.g., one mansion affects the average)
- **ML/DL Application**:
  - Handling skewed distributions in preprocessing (e.g., income data).
  - Imputing missing values in a way that isn't affected by extreme values.
  - Setting thresholds for outlier detection.
  - Used in feature scaling (e.g., **RobustScaler** in Python) to reduce the impact of outliers.

---

## 3 Mode (The "Most Common")

- **Definition**: The value that appears most frequently in the dataset.
- **Example**:
  - Shoe sizes: 7, 8, 8, 9, 10 → Mode = 8
- **Best Use Case**: For categorical data, such as:
  - Favorite color
  - Most sold product
- **ML/DL Application**:
  - Engineering features for categorical data.
  - Imputing missing values with the most common category.
  - Analyzing class imbalance in classification problems (e.g., identifying the most common class in an imbalanced dataset).

---

## When to Use Each Measure

| Measure   | Formula               | Best For               | Outlier Sensitivity | ML/DL Example                             |
|-----------|-----------------------|------------------------|---------------------|-------------------------------------------|
| **Mean**  | `Σx_i / n`            | Normal distributions   | High                | Calculating MSE in regression             |
| **Median**| Middle value          | Skewed data            | Low                 | Imputing missing values in skewed data    |
| **Mode**  | Most frequent value   | Categorical data       | None                | Handling class imbalance in classification|

---

## Example in Python

The following code demonstrates how to calculate the mean, median, and mode using Python libraries. It also shows how to use the median for scaling data with outliers.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Sample data with an outlier
housing_prices = [75000, 80000, 85000, 90000, 1500000]

# Calculate mean, median, and mode
print(f"Mean: {np.mean(housing_prices):.2f}")  # 357500.00
print(f"Median: {np.median(housing_prices)}")   # 85000.0
print(f"Mode: {pd.Series(housing_prices).mode()[0]}")  # 75000

# Use median for outlier-resistant scaling
scaler = RobustScaler(with_centering=True, with_scaling=True)
scaled_prices = scaler.fit_transform(np.array(housing_prices).reshape(-1, 1))
print(scaled_prices)
```

**Explanation**:
- The mean is heavily influenced by the outlier ($1,500,000), resulting in 357500.00, which doesn't represent most of the data.
- The median remains stable at $85,000, making it a better measure for skewed data.
- The mode is 75,000, as all values appear once except for the outlier.
- The `RobustScaler` uses the median to scale the data, making it less sensitive to outliers, which is useful in ML preprocessing.

---

## **Tools and Methods Summary**
- **Central Tendency**: `np.mean()`, `np.median()`, `statistics.mode()`, `pd.Series.mean()`.
- **Dispersion**: `np.var()`, `np.std()`, `np.percentile()`, `max() - min()`.
- **Shape**: `scipy.stats.skew()`, `scipy.stats.kurtosis()`.
- **Frequency**: `pd.value_counts()`.
- **Visualization**: `sns.histplot()`, `sns.boxplot()`, `sns.violinplot()`, `sns.lineplot()`.


---

# 2. Measures of Variability (Spread) 

---

## What is Variability?

Variability, or spread, measures how scattered data points are in a dataset. It tells you if the data clusters around the center (like the mean) or spreads out widely.

**Purpose**: To measure how spread out the data is, which helps in understanding datasets and making decisions in ML and DL.

**Why It’s Important in ML/DL**:
- Spots outliers that can confuse models.
- Helps scale features so algorithms like neural networks work better.
- Guides data preprocessing and model choices.

---

## 1. Range

**Definition**: The difference between the largest and smallest values in a dataset.

- **Formula**:  
  `Range = Maximum - Minimum`
- **Example**:  
  For house prices `[200, 250, 300, 350, 400]` (in thousands):  
  - Maximum: 400, Minimum: 200  
  - Range: `400 - 200 = 200` thousand dollars
- **Best Use**: A fast way to see the total spread of data.
- **ML/DL Use**:  
  - Checks for extreme values in features (e.g., very high prices).  
  - Identifies outliers that might affect model training.
- **Limitation**: Outliers can make the range misleading. For example, adding 1000 would change the range to 800, hiding the true spread.

**Key Point**: Range is quick but doesn’t show how data is distributed between the extremes.

---

## 2. Variance

**Definition**: The average of the squared differences between each data point and the mean.

- **Formula**:  
  `Variance = Σ(x_i - μ)² / n`  
  - `x_i`: each data point, `μ`: mean, `n`: number of points
- **Example**:  
  For house prices `[200, 250, 300, 350, 400]` (in thousands):  
  - Mean (`μ`): `(200 + 250 + 300 + 350 + 400) / 5 = 300`  
  - Squared differences: `(200-300)² = 10000`, `(250-300)² = 2500`, `(300-300)² = 0`, `(350-300)² = 2500`, `(400-300)² = 10000`  
  - Variance: `(10000 + 2500 + 0 + 2500 + 10000) / 5 = 5000` thousand dollars²
- **Best Use**: Good for precise spread measurement, especially with normal (bell-shaped) data.
- **ML/DL Use**:  
  - **PCA**: Finds important features by looking at variance.  
  - **Feature Selection**: Features with higher variance often matter more.  
  - **Loss Functions**: Used to measure errors in regression models.
- **Limitation**: Squared units (e.g., thousand dollars²) make it hard to interpret directly.

**Key Point**: Variance is exact but less intuitive due to squared units.

---

## 3. Standard Deviation (SD)

**Definition**: The square root of the variance, putting the spread back into the original units.

- **Formula**:  
  `SD = √Variance`
- **Example**:  
  For variance = 5000 thousand dollars²:  
  - SD: `√5000 ≈ 70.71` thousand dollars
- **Best Use**: When you need a clear, interpretable measure of spread.
- **ML/DL Use**:  
  - **Feature Scaling**: Standardizes features (mean = 0, SD = 1) for algorithms like gradient descent.  
  - **Model Evaluation**: Shows how much predictions vary.
- **Limitation**: Sensitive to outliers, like variance.

**Key Point**: SD is widely used because it’s easy to understand and matches the data’s units.

---

## 4. Mean Absolute Deviation (MAD)

**Definition**: The average of the absolute differences between each data point and the mean.

- **Formula**:  
  `MAD = Σ|x_i - μ| / n`
- **Example**:  
  For house prices `[200, 250, 300, 350, 400]` (in thousands):  
  - Mean (`μ`): 300  
  - Absolute differences: `|200-300| = 100`, `|250-300| = 50`, `|300-300| = 0`, `|350-300| = 50`, `|400-300| = 100`  
  - MAD: `(100 + 50 + 0 + 50 + 100) / 5 = 60` thousand dollars
- **Best Use**: When you want a measure less affected by outliers.
- **ML/DL Use**:  
  - **Robust Regression**: Helps models handle outliers (e.g., in Huber loss).  
  - **Anomaly Detection**: Finds unusual patterns, like fraud in transactions.
- **Limitation**: Less common in advanced stats compared to variance.

**Key Point**: MAD gives a simple average distance from the mean and resists outliers.

---

## 5. Interquartile Range (IQR)

**Definition**: The difference between the 75th percentile (Q3) and 25th percentile (Q1), showing the spread of the middle 50% of data.

- **Formula**:  
  `IQR = Q3 - Q1`
- **Example**:  
  For house prices `[200, 220, 250, 300, 350, 400, 450]` (in thousands):  
  - Q1 (25th percentile): 220  
  - Q3 (75th percentile): 400  
  - IQR: `400 - 220 = 180` thousand dollars
- **Best Use**: When you need a measure that ignores outliers and focuses on the central data.
- **ML/DL Use**:  
  - **Outlier Detection**: Marks values outside `Q1 - 1.5 × IQR` or `Q3 + 1.5 × IQR` as outliers.  
  - **Robust Scaling**: Used in tools like `RobustScaler` to make features less sensitive to outliers.
- **Limitation**: Only looks at the middle 50%, ignoring the tails.

**Key Point**: IQR is perfect for skewed data or when outliers are a concern.

---

## Comparison Table

| Measure            | Formula                   | Units         | Outlier Sensitivity | ML/DL Use                  |
|--------------------|---------------------------|---------------|---------------------|----------------------------|
| **Range**          | Max - Min                 | Same as data  | High                | Quick data checks          |
| **Variance**       | Σ(x_i - μ)² / n           | Squared units | High                | PCA, feature selection     |
| **Standard Deviation** | √Variance             | Same as data  | High                | Feature scaling            |
| **MAD**            | Σ|x_i - μ| / n            | Same as data  | Low                 | Robust models              |
| **IQR**            | Q3 - Q1                   | Same as data  | Low                 | Outlier detection          |

---

## When to Use Each Measure

- **Range**: For a fast look at spread.
- **Variance**: For advanced methods like PCA or regression.
- **Standard Deviation**: For most tasks, especially when units matter.
- **MAD**: When outliers might distort results.
- **IQR**: For skewed data or to avoid outlier impact.

The right choice depends on your data and ML/DL goals.

---

## Python Examples for ML/DL

### Example 1: Calculating Variability Measures

```python
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load California housing data (prices in USD)
data = fetch_california_housing()
prices = data.target * 100000  # Convert to dollars

# Calculate measures
print(f"Range: {np.ptp(prices):,.0f} USD")
print(f"Variance: {np.var(prices):,.0f} USD²")
print(f"SD: {np.std(prices):,.0f} USD")
print(f"MAD: {np.mean(np.abs(prices - np.mean(prices))):,.0f} USD")
print(f"IQR: {np.percentile(prices, 75) - np.percentile(prices, 25):,.0f} USD")
```

**Sample Output**:  
```
Range: 485,000 USD
Variance: 1,200,000,000 USD²
SD: 34,641 USD
MAD: 25,000 USD
IQR: 28,000 USD
```

This shows the spread of house prices, useful for regression tasks.

### Example 2: Robust Scaling with IQR

```python
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import fetch_california_housing

# Load California housing data
data = fetch_california_housing()
X = data.data  # Features

# Scale using IQR
scaler = RobustScaler(quantile_range=(25, 75))
scaled_features = scaler.fit_transform(X)

# Check IQR before and after for the first feature
feature = X[:, 0]
scaled_feature = scaled_features[:, 0]
print(f"Original IQR: {np.percentile(feature, 75) - np.percentile(feature, 25):.2f}")
print(f"Scaled IQR: {np.percentile(scaled_feature, 75) - np.percentile(scaled_feature, 25):.2f}")
```

**Sample Output**:  
```
Original IQR: 2.50
Scaled IQR: 1.00
```

Scaling with IQR reduces outlier effects, improving model performance.

---

## Summary

- **Range**: Fast and simple, but sensitive to outliers.
- **Variance**: Precise but in squared units.
- **Standard Deviation**: Clear and common in ML/DL.
- **MAD**: Robust to outliers, good for uneven data.
- **IQR**: Focuses on the middle 50%, resists outliers.

These measures help you understand your data, clean it, and build stronger ML/DL models.

---


# 3. Outliers and Leverage Points

In Machine Learning (ML) and Deep Learning (DL), data is the foundation of every model. However, not all data points are equal—some stand out and can disrupt how models learn. These are called **outliers** and **leverage points**. This guide explains what they are, why they matter, and how to handle them, using simple language, clear examples, and practical Python code.

---

## What is an Outlier?

An **outlier** is a data point that is very different from the rest of the dataset. It’s like the odd one out—much larger or smaller than the other values.

### Definition
An outlier is an unusual observation that doesn’t fit the general pattern of the data.

### Characteristics
- Outliers have extreme values in the **outcome** (the variable we predict, often called Y).
- They can affect key numbers like:
  - **Mean** (average): Outliers pull it up or down.
  - **Variance** (spread): Outliers make the data seem more scattered.
- They can be much higher or lower than typical values.

### Why Outliers Matter in ML
Outliers can cause trouble in ML models:
- **Lower accuracy**: They add **noise** or **bias**, leading to wrong predictions.
- **Skewed metrics**: They mess up measures like **mean squared error (MSE)**, which shows how well the model performs.
- **Training problems**: They can slow down or confuse training, especially in models sensitive to extremes, like linear regression or DL networks with small datasets.

### How to Find Outliers
Spotting outliers is a key step in preparing data. Here are common ways:
- **Visual Tools**:
  - **Boxplots**: Show outliers as points outside the "whiskers."
  - **Scatter plots**: Highlight points far from the main group.
  - **Histograms**: Show extreme values that don’t fit the pattern.
- **Statistical Methods**:
  - **Z-score**: If a point is more than 3 standard deviations from the mean, it’s likely an outlier.
  - **Interquartile Range (IQR)**: Points outside 1.5 times the IQR are outliers.
- **ML Tools**:
  - **Isolation Forest**: An algorithm that identifies outliers by isolating them in decision trees.
  - **DBSCAN (Density-Based Spatial Clustering)**: Marks sparse points as outliers.

### Example: Finding an Outlier with a Boxplot
Let’s look at exam scores: 55, 60, 65, 70, 75, 80, and 1000. The score 1000 looks out of place.

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
scores = [55, 60, 65, 70, 75, 80, 1000]

# Create a boxplot
plt.boxplot(scores)
plt.title("Boxplot of Exam Scores")
plt.show()
```

The boxplot shows **1000** as a point far above the rest, marking it as an outlier.

---

## Why Outliers Affect Machine Learning

Outliers can change how we understand data and how models behave. Let’s see how they impact key numbers.

### How Outliers Change Statistics
Here’s a table showing the effect:

| **Measure** | **Without Outlier** | **With Outlier**    |
|-------------|---------------------|---------------------|
| **Mean**    | Closer to the data  | Much higher/lower   |
| **Median**  | Stays the same      | Almost no change    |
| **Range**   | Smaller             | Much larger         |

**Example**: Salaries of $30,000, $35,000, $40,000, and $1,000,000:
- **Without $1,000,000**: Mean = $35,000, Median = $35,000, Range = $10,000.
- **With $1,000,000**: Mean = $276,250, Median = $35,000 (unchanged), Range = $970,000.

The mean jumps up a lot, but the median stays steady. This is why median-based methods are useful in ML when outliers exist.

---

## Handling Outliers

To make ML models work better, we need to deal with outliers. Here are three ways:

### 1. Remove Outliers
- If they’re **errors** (e.g., a typo), take them out.
- Example: If a height is listed as 20 feet, it’s likely a mistake and should be removed.

### 2. Transform the Data
- **Log Transformation**: Shrinks big values to lessen their impact.
  - Example: If salaries range from $10,000 to $1,000,000, applying `log(salary)` makes extreme values less dominant.
- **Normalization**: Scales all values (e.g., 0 to 1) to reduce outlier effects.
  - Example: Use Python’s `MinMaxScaler` for this.

### 3. Use Robust Models
- **Tree-based models**: Like Random Forests or Gradient Boosting, which handle outliers better than linear models.
- **Median-based methods**: Use measures like median absolute deviation (MAD) or IQR, which are less affected by outliers compared to mean-based methods.

### Example: Using Isolation Forest
Let’s detect an outlier with Python.

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Sample data
data = np.array([[10], [12], [15], [14], [11], [9000]])

# Train the model
model = IsolationForest(contamination=0.1)  # Expect 10% outliers
outliers = model.fit_predict(data)

print(outliers)  # Output: [1 1 1 1 1 -1]
```

Here, **9000** is flagged as an outlier (-1), while others are normal (1).

---

## What are Leverage Points?

A **leverage point** is a data point with an extreme value in the **input features** (often called X). It can pull the model toward it, especially in linear models.

### Definition
A leverage point is an observation with an extreme input that strongly influences the model’s fit.

### Characteristics
- Extreme in the **input** (X), not necessarily the outcome (Y).
- Changes the **slope** or direction of a model’s predictions.
- Has a big impact on how the model learns.

### Impact on ML Models
- **Distorts predictions**: Alters the model’s parameters (e.g., slope in regression).
- **Overfitting**: The model may focus too much on the leverage point.
- **Affects linear models most**: Tree-based models are less sensitive.

### Example: Leverage in House Prices
Let’s see how a leverage point affects a regression line.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = pd.DataFrame({
    'Size (sq.ft)': [1000, 1500, 2000, 10000],  # 10000 is extreme
    'Price ($)': [200000, 300000, 400000, 1000000]
})

# Plot regression
sns.lmplot(x='Size (sq.ft)', y='Price ($)', data=data)
plt.title("Regression with Leverage Point")
plt.show()
```

The house with **10,000 sq. ft.** pulls the line up, changing predictions for smaller houses.

---

## Outliers vs. Leverage Points

Here’s how they differ:

| **Aspect**       | **Outlier**                        | **Leverage Point**                |
|------------------|------------------------------------|-----------------------------------|
| **Location**     | Extreme in **outcome** (Y)         | Extreme in **input** (X)          |
| **Impact**       | Changes stats like mean and variance | Pulls the model's line or curve   |
| **Example**      | A CEO’s $1 salary in a salary dataset | A 100-year-old student in a class |

### High-Leverage Outliers
A point can be both if it’s extreme in X and Y. Example: A **10,000 sq. ft. house sold for $1**. It pulls the model and skews stats.

---

## Summary

- **Outliers**:
  - Extreme values in the **outcome** (Y).
  - Example: A billionaire’s income in a salary dataset.
  - Impact: Skews statistical measures like mean and variance, reduces model accuracy.
- **Leverage Points**:
  - Extreme values in the **input** (X).
  - Example: A much older student in a class dataset.
  - Impact: Pulls the model's line or curve, increases overfitting risk.
- **High-Leverage Outliers**:
  - Extreme in both X and Y.
  - Example: A 10,000 sq. ft. house sold for $1.
  - Impact: Strongly affects model fit and accuracy.
- **Handling Strategies**:
  - **Remove**: Eliminate outliers or leverage points if they are errors.
  - **Transform**: Use log transformation, normalization, or other scaling methods.
  - **Robust Models**: Use median-based metrics or tree-based algorithms (e.g., Random Forests) to reduce sensitivity to extreme values.

By managing outliers and leverage points, you can build stronger ML and DL models that handle real-world data better.

--- 

# **4. 📊 Five Number Summary**

---

## **Introduction to Five Number Summary**

The **Five Number Summary** is a simple way to describe a dataset using five key values:

1. **Minimum**: The smallest value in the dataset.
2. **First Quartile (Q1)**: The value where 25% of the data falls below (25th percentile).
3. **Median (Q2)**: The middle value, where 50% of the data is below and 50% is above (50th percentile).
4. **Third Quartile (Q3)**: The value where 75% of the data falls below (75th percentile).
5. **Maximum**: The largest value in the dataset.

These five numbers give a quick snapshot of how the data is spread out and where its center lies. They are especially useful in data analysis for ML and DL, where understanding the data's range and distribution is a critical first step.

> **Why it matters**: Imagine you're analyzing house prices. These five numbers can tell you the cheapest and most expensive homes, the typical price range, and whether some prices stand out as unusual.

---

## **Core Components and Their ML/DL Relevance**

### **1. Minimum and Maximum**
- **What they do**: Show the full range of the data (from lowest to highest).
- **In ML**: Used in feature scaling (e.g., Min-Max scaling) to adjust data to a range like [0, 1].
- **In DL**: Help choose activation functions (e.g., ReLU or sigmoid), which work best with specific input ranges.

### **2. Quartiles (Q1, Q2, Q3)**
- **What they do**: Split the data into four parts, showing how it’s distributed.
  - *Percentiles* divide data into 100 equal parts. Q1 is the 25th percentile, meaning 25% of values are below it, and so on.
- **In ML**: Used to bin data into categories or for robust scaling that ignores extreme values.
- **In DL**: Guide normalization layers (e.g., batch normalization) to keep training stable.

### **3. Interquartile Range (IQR)**
- **Formula**: \( \text{IQR} = Q3 - Q1 \)
- **What it does**: Measures the spread of the middle 50% of the data.
- **In ML**: Identifies outliers—values far outside the norm—that might confuse a model.
- **In DL**: Helps design loss functions that aren’t thrown off by outliers.

---

## **How to Calculate It**

Here’s how to find the Five Number Summary:

1. **Sort the data** from smallest to largest.
2. **Find the minimum** (first value) and **maximum** (last value).
3. **Find the median (Q2)**: The middle value. If the dataset has an even number of points, average the two middle values.
4. **Find Q1**: The median of the lower half (values below Q2).
5. **Find Q3**: The median of the upper half (values above Q2).

### **Python Example**
Let’s calculate it using the California Housing dataset:

```python
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load the dataset
data = fetch_california_housing()
prices = data.target * 1000  # Convert to USD for readability

# Calculate the Five Number Summary
minimum = np.min(prices)
Q1 = np.percentile(prices, 25)
median = np.median(prices)
Q3 = np.percentile(prices, 75)
maximum = np.max(prices)

# Display the results
print(f"Minimum: ${minimum:,.0f}")
print(f"Q1: ${Q1:,.0f}")
print(f"Median: ${median:,.0f}")
print(f"Q3: ${Q3:,.0f}")
print(f"Maximum: ${maximum:,.0f}")
```

**Output** (example values):
```
Minimum: $149,000
Q1: $1,290,000
Median: $1,799,000
Q3: $2,641,000
Maximum: $5,000,000
```

This code shows the range and distribution of house prices in California.

---

## **Why It’s Useful**

The Five Number Summary is valuable because it:
- **Summarizes data quickly**: Gives a clear picture of spread and center.
- **Spots outliers**: Highlights unusual values that might need attention.
- **Supports visualization**: Forms the basis of box plots.

> **Fun fact**: Unlike the average (mean), the median isn’t swayed by extreme values, making it a sturdy anchor for uneven data—like house prices with a few mansions.

---

## **Visualizing with a Box Plot**

A **box plot** turns the Five Number Summary into a picture:
- **Box**: From Q1 to Q3 (the IQR, covering the middle 50% of data).
- **Line in the box**: The median (Q2).
- **Whiskers**: Lines stretching to the minimum and maximum (unless there are outliers).
- **Outliers**: Points outside the whiskers, plotted separately.

### **Python Visualization**
```python
import matplotlib.pyplot as plt

# Create a box plot
plt.boxplot(prices, vert=False)
plt.title("California Housing Prices Distribution")
plt.xlabel("Price (USD)")
plt.show()
```

This code generates a horizontal box plot. When you run it in a Jupyter notebook, you’ll see the plot directly below the cell, showing the data’s spread and any outliers.

> **Think of it like this**: The box plot is a snapshot of your data’s “personality”—where it clusters, how far it stretches, and which values are the rebels standing apart.

---

## **Identifying Outliers**

Outliers are extreme values that don’t fit with the rest. To find them:
- Calculate the IQR: \( Q3 - Q1 \).
- Define bounds:
  - Lower bound: \( Q1 - 1.5 \times \text{IQR} \)
  - Upper bound: \( Q3 + 1.5 \times \text{IQR} \)
- Any value **below the lower bound** or **above the upper bound** is an outlier.

### **Python Example**
```python
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
print(f"Number of outliers: {len(outliers)}")
```

This identifies houses with unusually low or high prices.

> **Why care?**: Outliers are like noisy neighbors—they can distract your ML model from the real patterns.

---

## **Advanced ML/DL Applications**

### **1. Outlier Detection**
- **Use**: Clean data by finding and handling outliers.
- **Code** (from above):
```python
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
```

### **2. Robust Feature Scaling**
- **Use**: Scale data using the IQR to reduce outlier impact.
- **Formula**: \( \text{Scaled Value} = \frac{x - \text{Median}}{\text{IQR}} \)
- **Code**:
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
```

### **3. Data Binning**
- **Use**: Turn numbers into categories (e.g., “Low,” “High”) using quartiles.
- **Code**:
```python
import pandas as pd

bins = [minimum, Q1, median, Q3, maximum]
labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
price_groups = pd.cut(prices, bins=bins, labels=labels)
print(price_groups[:5])  # Show first 5 categories
```

These techniques make data ready for ML models like decision trees or DL networks like neural networks.

---

## **Comparative Analysis**

| **Metric**         | **ML Application**               | **DL Consideration**            |
|---------------------|----------------------------------|---------------------------------|
| **Minimum/Maximum** | Sets scaling limits             | Guides activation ranges       |
| **Quartiles**       | Bins data into categories       | Helps normalize layers         |
| **IQR**             | Spots outliers                  | Builds robust loss functions   |
| **Median**          | Fills missing values            | Centers batch normalization    |

This table shows how the Five Number Summary powers both ML and DL workflows.

---

## **Summary**

- The **Five Number Summary** offers a fast, clear view of your data’s range and distribution.
- It’s perfect for spotting outliers and understanding patterns, especially with box plots.
- In ML and DL, it helps prepare data—scaling features, handling outliers, and more—to build better models.

---


# 5. Percentiles and Quartiles

In data analysis, **percentiles** and **quartiles** help us understand how data is spread out. They’re powerful tools in Machine Learning (ML) for preparing data, spotting unusual values, and checking how well models perform. Let’s explore these concepts step by step.

---

## 1. Percentiles in Machine Learning

### What is a Percentage?
A **percentage** shows how much of something you have out of 100.  
- **Example**: If you score 24 out of 30 on a test:  
  ```
  (24 ÷ 30) × 100 = 80%
  ```  
  This means you got 80% of the test right.  
- **Analogy**: Picture a pizza cut into 100 slices. Eating 80 slices means you ate 80% of the pizza.

### What is a Percentile?
A **percentile** tells you your position compared to others. It shows what percentage of values are **below** yours.  
- **Example**: If your score is in the **90th percentile**, you did better than 90% of people.  
- **Analogy**: In a race with 100 runners, being in the 90th percentile means you beat 90 of them.

### How Percentiles Help in ML
Percentiles are used in ML for:  
- **Feature Scaling**: Adjusting data (e.g., Min-Max scaling) so it fits between 0 and 1.  
- **Anomaly Detection**: Finding outliers, like values above the 95th percentile.  
- **Model Evaluation**: Checking how a model performs across different parts of the data.

---

## 2. Quartiles in Machine Learning

### What are Quartiles?
**Quartiles** split your data into **four equal parts**:  
- **Q1 (25th percentile)**: 25% of the data is below this point.  
- **Q2 (50th percentile)**: The **median**, where 50% of the data sits below.  
- **Q3 (75th percentile)**: 75% of the data is below this value.  

**Example**: For the scores `[10, 20, 30, 40, 50, 60, 70, 80]`:  
- **Q1**: 25  
- **Q2 (Median)**: 45  
- **Q3**: 65  
- **Analogy**: Imagine slicing a cake into four equal pieces. Each piece is 25% of the cake.

### How Quartiles Help in ML
Quartiles are useful for:  
- **Data Visualization**: Box plots use quartiles to show how data is spread and highlight outliers.  
- **Feature Engineering**: Grouping data into categories (e.g., "Low," "Medium," "High") based on quartiles.  
- **Decision Trees**: Setting split points in decision-making algorithms.

### Python Example
```python
import numpy as np
scores = [55, 68, 72, 75, 82, 88, 95]

# Calculate quartiles
Q1 = np.percentile(scores, 25)  # 68.0
median = np.median(scores)      # 75.0
Q3 = np.percentile(scores, 75)  # 88.0

print(f"Q1: {Q1}")
print(f"Median: {median}")
print(f"Q3: {Q3}")
```

---

## 3. Interquartile Range (IQR) in Machine Learning

### What is IQR?
The **Interquartile Range (IQR)** measures the spread of the **middle 50%** of your data.  
- **Formula**:  
  ```
  IQR = Q3 - Q1
  ```  
- **Example**: For scores `[50, 60, 70, 80, 90]`:  
  - **Q1**: 60  
  - **Q3**: 80  
  - **IQR**: 80 - 60 = 20  
- **Analogy**: Think of a pizza cut into four parts. The IQR is the size of the two middle slices—the part you focus on most.

### Why IQR Matters in ML
- **Outlier Detection**: Values outside **Q1 - 1.5 × IQR** or **Q3 + 1.5 × IQR** are often outliers.  
- **Model Robustness**: By focusing on the middle 50%, IQR helps models ignore extreme values.

---

## Practical Uses in Machine Learning

### 1. Handling Missing Data
When data is missing, the **median (Q2)** is a great choice to fill gaps, especially if the data has outliers.  
- **Example (Titanic Dataset)**:  
  ```python
  import pandas as pd
  titanic = pd.read_csv("titanic.csv")
  median_age = titanic['Age'].median()
  titanic['Age'].fillna(median_age, inplace=True)
  ```

### 2. Feature Engineering with Quartiles
Quartiles can turn numbers into groups for better analysis.  
- **Example**: Grouping ages into "Young," "Adult," and "Senior":  
  ```python
  titanic['Age_Group'] = pd.qcut(titanic['Age'], q=[0, 0.25, 0.75, 1], labels=['Young', 'Adult', 'Senior'])
  ```

### 3. Removing Outliers with IQR
Use IQR to clean data by removing extreme values.  
- **Example**:  
  ```python
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  titanic_cleaned = titanic[(titanic['Age'] >= lower_bound) & (titanic['Age'] <= upper_bound)]
  ```

---

## Summary Table

| **Concept**    | **Definition**                          | **ML Use**                              |
|----------------|-----------------------------------------|-----------------------------------------|
| **Percentile** | Shows your rank compared to others      | Model evaluation, anomaly detection     |
| **Quartiles**  | Splits data into 4 equal parts          | Feature engineering, decision trees     |
| **IQR**        | Spread of the middle 50% of data        | Outlier removal, data preprocessing     |

---

## Key Takeaways
- **Percentiles** tell you where a value ranks—great for spotting outliers or scaling data.  
- **Quartiles** divide data into four parts, helping with visualization and grouping.  
- **IQR** focuses on the middle of the data, making models more reliable by ignoring extremes.  
- These tools simplify data preparation and improve ML models in real-world tasks.

--- 

# 6. Measures of Shape

---

## What Are Measures of Shape?
Measures of shape describe the **pattern** of your data’s distribution. They answer two main questions:
- **Is the data uneven or lopsided?** (That’s **skewness**.)
- **Is the data very peaked or flat, with lots of outliers?** (That’s **kurtosis**.)

---

## Symmetric Distribution (No Skew)
A **symmetric** distribution looks the same on both sides when split down the middle.
- **Features**:
  - The left and right halves are like mirror images.
  - The **mean** (average), **median** (middle value), and **mode** (most common value) are about the same.
  - A histogram looks balanced, like a bell curve.
- **Example**: Heights of adults often form a symmetric “bell” shape.

---

## Skewed Distribution (Lopsided Data)
**Skewness** shows if the data leans to one side. It measures how uneven the distribution is.

### 1. Right-Skewed (Positive Skewness)
- **What it looks like**: Most values are small (on the left), with a few large ones stretching the tail to the right.
- **Order**: Mode < Median < Mean.
- **Example**: Income data—most people earn average amounts, but a few millionaires pull the tail right.
- **Picture it**: Like a slide sloping down to the right.

### 2. Left-Skewed (Negative Skewness)
- **What it looks like**: Most values are large (on the right), with a few small ones stretching the tail to the left.
- **Order**: Mode > Median > Mean.
- **Example**: Exam scores—most students score high, but a few score very low.

---

## Spotting Skewness with Box Plots
A **box plot** is a quick way to see skewness:
- **Right-Skewed**: The right side of the box is bigger, and the right “whisker” (line) is longer.
- **Left-Skewed**: The left side is bigger, and the left whisker is longer.
- **Symmetric**: Both sides and whiskers look equal.

---

## Why Skewness Matters in ML and DL
Skewed data can cause problems for models. Here’s how:

| **Area**             | **Why It’s a Problem**                                      |
|----------------------|------------------------------------------------------------|
| **Training Models**  | Linear regression expects normal data—skewness can mess it up. Neural networks may train slower with skewed inputs. |
| **Metrics**          | In classification, accuracy fails if classes are skewed (unbalanced). Use F1-score instead. In regression, skewed data affects Mean Squared Error (MSE). |
| **Outliers**         | Skewed data can hide unusual points or make normal ones look odd. |

### Fixing Skewness
Here are ways to handle skewness:
- **Log Transform**: Shrinks big values (great for right-skewed data like prices).
- **Yeo-Johnson Transform**: Adjusts data to look more normal.
- **Batch Normalization**: Fixes skewed inputs in neural networks.
- **Robust Scaling**: Uses the median instead of the mean to ignore outliers.

### When Skewness Isn’t a Big Deal
Some models don’t mind skewness:
- **Tree-based models** (e.g., Random Forest) work fine with lopsided data.
- **Decision Trees** don’t need normal data.

---

## Kurtosis: Peaks and Tails
**Kurtosis** tells us how **sharp** or **flat** the data’s peak is and how many extreme values (outliers) it has.

### Types of Kurtosis
- **Leptokurtic (High Kurtosis)**:
  - Sharp peak, thick tails (lots of outliers).
  - **Example**: Stock prices—mostly small changes, but big jumps happen sometimes.
  - **Visual**: A tall, pointy hill.
- **Platykurtic (Low Kurtosis)**:
  - Flat peak, thin tails (few outliers).
  - **Example**: Adult shoe sizes—most are average, with rare extremes.
  - **Visual**: A wide, gentle hill.
- **Mesokurtic**:
  - Normal curve with a medium peak and tails.

---

## Why Skewness and Kurtosis Matter
- **Skewness**: Shows if data leans one way, which can bias predictions.
- **Kurtosis**: Highlights extreme values, important for tasks like risk analysis.

---

## Checking Skewness and Kurtosis in Python
Here’s a simple example to calculate and visualize them:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create fake normal data
np.random.seed(42)
data = np.random.normal(0, 100, 1000)
df = pd.DataFrame({'x': data})

# Calculate skewness and kurtosis
print(f"Skewness: {df['x'].skew():.2f}")
print(f"Kurtosis: {df['x'].kurt():.2f}")

# Plot it
sns.histplot(df['x'], kde=True, bins=30)
plt.title("Data Distribution")
plt.show()
```
- **Output**: For normal data, skewness is near 0, and kurtosis is near 0.
- **Graph**: Shows a bell curve if symmetric.

---

## Fixing Skewed Data in Python
To make skewed data more normal:

```python
from sklearn.preprocessing import PowerTransformer

# Use Yeo-Johnson to fix skewness
pt = PowerTransformer(method='yeo-johnson')
df['x_fixed'] = pt.fit_transform(df[['x']])

# Check new skewness
print(f"New Skewness: {df['x_fixed'].skew():.2f}")
```

---

## Real-World Uses in ML and DL
### 1. Batch Normalization (Neural Networks)
- **What it does**: Keeps layer inputs balanced, even if data is skewed.
- **How it works**: Adjusts data using the mean and variance of each batch.
- **Code**:
  ```python
  from tensorflow.keras.layers import BatchNormalization
  model.add(BatchNormalization())
  ```

### 2. NLP Example
- Word frequencies are often right-skewed. Use **TF-IDF** to balance them for text models.

### 3. Finance Example
- Stock returns are skewed and have high kurtosis. Log-transform them for better predictions.

---

## Key Points
- **Skewness** shows if data is uneven, which can hurt model accuracy.
- **Kurtosis** tells you about peaks and outliers, useful for spotting extremes.
- Use **transforms** (log, Yeo-Johnson) or **normalization** to fix skewness.
- **Visualize** with histograms or box plots to see the shape clearly.

---

# 7. Frequency and Cumulative Distributions

Understanding how data is distributed is essential for building effective Machine Learning (ML) models. **Frequency distributions** and **cumulative frequency distributions** are simple yet powerful tools to summarize data, spot patterns, and prepare it for training.

---

## 1. Frequency Distribution

### What is it?
A **frequency distribution** counts how many times each value appears in a dataset. It organizes data into a table or chart to show patterns easily.

### Example
Suppose you ask 10 people, *"How many pets do you own?"* Their answers are: `[0, 1, 1, 2, 0, 3, 1, 0, 2, 1]`.

Here’s the frequency distribution:

| Number of Pets | Frequency (Count) |
|----------------|-------------------|
| 0              | 3                 |
| 1              | 4                 |
| 2              | 2                 |
| 3              | 1                 |

This shows that most people (4 out of 10) own 1 pet.

### Why It Matters
- **Identifies patterns**: Highlights common or rare values.
- **Real-world use**: Used in education (e.g., grade tracking), business (e.g., sales analysis), and research.
- **In ML**: Helps understand data before feeding it into a model.

### Python Code
```python
import pandas as pd

data = [0, 1, 1, 2, 0, 3, 1, 0, 2, 1]
freq_dist = pd.Series(data).value_counts().sort_index()
print(freq_dist)
```

---

## 2. Cumulative Frequency Distribution

### What is it?
A **cumulative frequency distribution** adds up the frequencies step-by-step, showing the total count up to each value. It answers questions like, *"How many people own up to a certain number of pets?"*

### Example
Using the same pet data:

| Number of Pets | Frequency | Cumulative Frequency |
|----------------|-----------|-----------------------|
| 0              | 3         | 3                     |
| 1              | 4         | 3 + 4 = 7             |
| 2              | 2         | 7 + 2 = 9             |
| 3              | 1         | 9 + 1 = 10            |

From this, 9 people own 2 pets or fewer.

### Why It Matters
- **Percentiles**: Helps find thresholds (e.g., top 25% of values).
- **Data prep**: Useful for normalizing features in ML.
- **Quick insights**: Answers cumulative questions efficiently.

### Python Code
```python
cumulative_dist = freq_dist.cumsum()
print(cumulative_dist)
```

---

## Key Differences

| **Aspect**       | **Frequency Distribution**           | **Cumulative Frequency Distribution** |
|------------------|--------------------------------------|---------------------------------------|
| **Definition**   | Counts each value’s occurrences.     | Running total of frequencies.         |
| **Purpose**      | Shows individual value counts.       | Shows totals up to each value.        |
| **Example Question** | "How many own exactly 2 pets?" (2) | "How many own 2 pets or fewer?" (9) |

---

## Visualizing Distributions

### 1. Histograms
A **histogram** is a bar chart showing the frequency of values. It helps visualize the data’s shape and detect outliers.

### 2. Box Plots
A **box plot** shows the median, quartiles, and outliers, giving a snapshot of data spread.

### ML Use
- **Exploratory Data Analysis (EDA)**: Understand data before modeling.
- **Feature Engineering**: Identify rare values for better features.
- **Outlier Detection**: Decide which values to adjust or remove.

---

## Practical Applications in Machine Learning

### 1. Checking Class Imbalance
In classification tasks, uneven class sizes can reduce model performance. Frequency distributions help spot this issue.

**Example (Iris Dataset)**:
```python
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
target_counts = pd.Series(iris.target).value_counts()
target_counts.plot(kind='bar')
plt.title("Class Distribution in Iris Dataset")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
```

### 2. Grouping Data with Quartiles
Cumulative distributions help divide data into groups (e.g., low, medium, high) based on percentiles.

**Example**:
```python
import numpy as np

data = [0, 1, 1, 2, 0, 3, 1, 0, 2, 1]
quartiles = np.percentile(data, [25, 50, 75])
print(f"Quartiles: {quartiles}")
```

---

## Advanced ML Applications

### 1. Balancing Data
If some classes are rare, tools like **SMOTE** can create synthetic samples to balance the dataset.

**Code**:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 2. Setting Classification Thresholds
Cumulative distributions help choose optimal cutoff points for classifying data.

**Code**:
```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
optimal_idx = np.argmax(precision + recall)
optimal_threshold = thresholds[optimal_idx]
```

---

## Key Takeaways
- **Frequency distributions** count how often values appear, revealing patterns.
- **Cumulative frequency distributions** track running totals, aiding in percentile analysis.
- **Visualization** (histograms, box plots) makes data easier to interpret.
- **ML benefits**: Improves data preprocessing, feature engineering, and model performance.

---

# 8. Normal Distribution, Scaling, and Their Role

---

## Introduction to Normal Distribution

The **normal distribution**, also called the Gaussian distribution, is a bell-shaped curve that shows how data points are spread around the mean (average). It’s a key concept in ML because many algorithms rely on its properties.

### Key Properties 
- **Symmetry**: The curve is the same on both sides of the mean.
- **Mean, Median, Mode**: These are all equal and sit at the center.
- **Most Data Near the Mean**: Values cluster around the average, with fewer points further away.

### The 68-95-99.7 Rule (Empirical Rule)
- About **68%** of data is within **1 standard deviation (σ)** of the mean.
- About **95%** is within **2σ**.
- About **99.7%** is within **3σ**.

### Why It Matters in ML
- **Linear Regression**: Assumes errors (differences between predicted and actual values) are normally distributed.
- **Deep Learning**: Neural network weights are often set initially using values from a normal distribution.
- **Statistics**: Helps calculate probabilities and make predictions.

---

## Standard Normal Distribution

The **standard normal distribution** is a simplified version of the normal distribution with:
- **Mean (μ) = 0**
- **Standard Deviation (σ) = 1**

It’s used to standardize data, making it easier to compare values from different datasets.

### Z-Scores
A **z-score** shows how far a data point is from the mean in terms of standard deviations.

**Formula**:
\[
z = \frac{X - \mu}{\sigma}
\]
- \(X\): The data point
- \(\mu\): Mean
- \(\sigma\): Standard deviation

**Example**:
- If a test score is 85, with \(\mu = 75\) and \(\sigma = 5\):
  \[
  z = \frac{85 - 75}{5} = 2
  \]
  This score is 2 standard deviations above the mean.

### Why It’s Useful
- Compares data across different scales (e.g., test scores vs. heights).
- Used in ML for **hypothesis testing** and **confidence intervals**.
- Helps standardize features for algorithms.

---

## Scaling in Machine Learning

**Scaling** adjusts the range of data features so they contribute equally to a model. Without scaling, features with larger ranges (e.g., income in dollars) might overpower smaller ones (e.g., age in years).

### Why Scale Data?
- **Fairness**: Ensures all features have equal influence.
- **Better Performance**: Helps algorithms like K-Nearest Neighbors (KNN) and neural networks work faster and more accurately.
- **Distance-Based Models**: Scaling is critical for methods that measure distances between points.

---

## Core Scaling Methods

Here are the main scaling techniques, their uses, and ML/DL examples:

### 1. Standardization (Z-Score Scaling)
- **What It Does**: Sets the mean to 0 and standard deviation to 1.
- **Formula**: \(z = \frac{X - \mu}{\sigma}\)
- **When to Use**: For normally distributed data or features with different units.
- **ML Examples**: 
  - Principal Component Analysis (PCA)
  - Support Vector Machines (SVM)

**Python Code**:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 2. Min-Max Scaling
- **What It Does**: Rescales data to a range, usually [0, 1].
- **Formula**: 
  \[
  X' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
  \]
- **When to Use**: For algorithms needing bounded inputs (e.g., neural networks).
- **ML Examples**: 
  - Image processing (pixel values from 0-255 to 0-1)
  - KNN

**Python Code**:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)
```

### 3. Robust Scaling
- **What It Does**: Uses the median and interquartile range (IQR) to scale, reducing the impact of outliers.
- **Formula**: 
  \[
  X' = \frac{X - \text{median}}{\text{IQR}}
  \]
- **When to Use**: For data with outliers.
- **ML Examples**: 
  - Regression with noisy data

**Python Code**:
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 4. Normalization (L2 Norm Scaling)
- **What It Does**: Adjusts data so each point has a unit length.
- **Why**: Ensures all data points contribute equally, even if their raw sizes differ.  
- **Formula**: 
  \[
  X' = \frac{X}{\|X\|}
  \]
  (where \(\|X\|\) is the Euclidean norm)
- **When to Use**: For text data or clustering.
- **ML Examples**: 
  - Text processing (TF-IDF vectors)
  - k-means clustering

**Python Code**:
```python
from sklearn.preprocessing import Normalizer
scaler = Normalizer(norm='l2')
X_scaled = scaler.fit_transform(X_train)
```

### Comparison Table
| Method             | Formula                          | Best For                  |
|--------------------|----------------------------------|---------------------------|
| Standardization    | \(z = \frac{X - \mu}{\sigma}\)  | Normal data, PCA, SVM     |
| Min-Max Scaling    | \(X' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}\) | Neural networks, KNN      |
| Robust Scaling     | \(X' = \frac{X - \text{median}}{\text{IQR}}\) | Outliers, regression      |
| Normalization      | \(X' = \frac{X}{\|X\|}\)        | Text, clustering          |

---

## When Scaling Isn’t Needed
- **Decision Trees** and **Random Forests**: These don’t rely on distances or scales.
- When all features are already in the same units (e.g., height and width in meters).

---

## Practical Example

Here’s how to generate and scale data in Python:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Generate normal distribution data
data = np.random.normal(0, 1, 1000)

# Plot it
plt.hist(data, bins=30, density=True, alpha=0.6, color='blue')
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# Standardize
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data.reshape(-1, 1))

# Min-Max Scale
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
data_minmax = minmax_scaler.fit_transform(data.reshape(-1, 1))
```

This code:
- Creates a normal distribution.
- Visualizes it with a histogram.
- Applies two scaling methods.

---

## Advanced Topics in Deep Learning

### 1. Batch Normalization
- **What It Does**: Normalizes layer inputs during training to stabilize learning.
- **Formula**: 
  \[
  \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta
  \]
  (where \(\mu_B\) and \(\sigma_B\) are batch statistics, and \(\gamma\), \(\beta\) are learned)
- **Benefit**: Speeds up training and reduces sensitivity to initial weights.
- **Use**: In convolutional neural networks (CNNs).

**Python Code**:
```python
from tensorflow.keras.layers import BatchNormalization
model.add(BatchNormalization())
```

### 2. Layer Normalization
- **What It Does**: Normalizes across features for each data point.
- **Use**: Common in Transformers and recurrent neural networks (RNNs).
- **Benefit**: Handles varying input sizes effectively.

**Python Code**:
```python
from tensorflow.keras.layers import LayerNormalization
model.add(LayerNormalization())
```

---

## Key Takeaways
- **Normal Distribution**: A bell-shaped curve that’s central to ML assumptions.
- **Z-Scores**: Standardize data for fair comparison.
- **Scaling**: Adjusts data ranges to improve model performance.
- **Methods**: Standardization, Min-Max, Robust, and Normalization suit different needs.
- **Deep Learning**: Batch and Layer Normalization stabilize training.
- **When to Skip**: Scaling isn’t needed for tree-based models.

---

