
# ML: Types of Models

---

## Understanding Features & Labels

### What are Features?
Features are the **columns in a dataset** that describe the data. They represent the characteristics or properties of the data.

#### Examples of Features:
- Age
- Gender
- Experience
- Salary

### What is a Label?
A **label** is the output we get after training the model. It is the value we aim to predict.

#### Example:
- If we want to **predict the salary** of an employee with **6 years of experience**, we train a model.
- The model predicts a **salary value** → This is the **label**.

#### Real-World Label Examples:
| Scenario | Label |
|----------|--------|
| Predicting if a pet is a **cat or dog** | Pet Type |
| Determining if a pet is **sick or healthy** | Health Condition |
| Estimating the **age of a pet** | Age |

---

## Labeled vs. Unlabeled Data

### Labeled Data:
- Comes with a **tag or label**, like a name, type, or number.
- Example: A dataset where each image of a dog is labeled as **“Dog”**.

### Unlabeled Data:
- Has **no predefined labels**.
- Example: A dataset of pet images **without names or types**.

---

## Types of Models

### Supervised Learning
- **Uses both input features & labels** to train the model.
- Commonly used in **real-world applications**.
- Supervised learning involves training a model using labeled data, where both input and output variables are provided. The model learns to map inputs to outputs.

#### Formula:
```math
Supervised Learning = Features + Labels
```

#### Examples of Supervised Learning:
- Predicting the **salary** of an employee based on **experience** (input feature) and using historical salary data (label).
- **Image Recognition**
- **Text Processing**
- **Recommendation Systems**

### Unsupervised Learning
- **Uses only input features** (No labels provided).
- Groups the data based on **similarities**.

#### Purpose:
```math
Unsupervised Learning = Features (No Labels)
```

#### Example of Unsupervised Learning:
- Grouping data points based on their **similarities** without knowing the labels.
- **Grouping customers based on shopping behavior.**
- **Finding similar news articles automatically.**

---

## Types of Supervised Learning Models

### Regression Models
**Regression models** predict continuous numerical values.

#### Examples:
- **Weight of an animal**
- **Employee salary**
- **Stock market prices**
- **Predicting house prices**

#### Algorithms:
- Linear Regression
- Decision Trees (Regression)
- Support Vector Regression (SVR)
- Neural Networks

### Classification Models
Used to **predict categories or labels**.
- **Purpose**: Assign inputs to **discrete categories (classes)**.

#### Examples:
- **Type of pet (Dog or Cat)**
- **Gender classification (Male or Female)**
- Spam detection (spam vs. not spam).
- Diagnosing diseases (positive vs. negative).
- **Taste classification (Good, Bad, Not Good)**

#### Algorithms:
- Logistic Regression
- Support Vector Machines (SVM)
- k-Nearest Neighbors (k-NN)
- Decision Trees (Classification)
- Naive Bayes
- Neural Networks

| Type | Regression | Classification |
|------|-----------|---------------|
| **Prediction Type** | Continuous Value | Categorical Value |
| **Examples** | Salary, Stock Prices | Dog vs. Cat, Spam vs. Ham |

---

## Types of Unsupervised Learning Models
Unsupervised learning involves training a model on unlabeled data. The goal is to find patterns or structures in the data.

### Types of Unsupervised Learning:

### 🔹 Clustering
- The process of grouping data into different clusters based on similarity.
- **Goal:** To categorize data points into meaningful clusters.
- **Goal**: To predict continuous values like test scores, or group similar items together.
- Example: Customer segmentation in marketing.

#### 🔸 Types of Clustering
1. **Partitional Clustering:** Each data point belongs to a single cluster.
2. **Hierarchical Clustering:** Clusters within clusters; a data point may belong to multiple clusters.

#### 🔸 Popular Clustering Algorithms:
- **K-Means**: A method that partitions data into K clusters based on similarity.
- **Expectation Maximization**: An algorithm that iteratively adjusts cluster parameters.
- **Hierarchical Cluster Analysis (HCA)**: Builds a hierarchy of clusters.

### 🔹 Association Rule Learning
- Attempts to find relationships or patterns between different entities.
- Example: Market Basket Analysis (e.g., Amazon recommending frequently bought-together products).


### Applications of Unsupervised Learning:
- **Airbnb**: Clustering users for personalized recommendations.
- **Amazon**: Recommending frequently bought items together.
- **Credit Card Fraud Detection**: Identifying unusual patterns or behaviors in transactions.

### Dimensionality Reduction
- **Reduces the number of features** while retaining important information.
- Helps in **data visualization** and **speeding up models**.
- Example: **Reducing a high-dimensional dataset while preserving relationships**.

---

## 3️ Reinforcement Learning
Reinforcement learning enables an agent to learn by interacting with an environment. It learns by trial and error and optimizes actions based on rewards.

### 🔹 Key Concepts
- **Agent:** The learner or decision-maker.
- **Environment:** The world in which the agent operates.
- **Actions:** Possible moves the agent can take.
- **Reward & Punishment:** Signals for positive and negative behavior.

### 🔹 Goal of Reinforcement Learning
- Find a model that maximizes cumulative rewards over time.
- Example: An AI playing chess tries to maximize points won over multiple moves.

---

## Summary of Learning Types:

| Learning Type         | Key Characteristics                                        | Goal                                             |
|-----------------------|------------------------------------------------------------|--------------------------------------------------|
| **Supervised Learning** | Uses labeled data, learns from input-output pairs          | Predict labels for new, unseen data              |
| **Unsupervised Learning** | No labeled data, finds patterns or groupings in data      | Discover patterns and relationships in data      |
| **Reinforcement Learning** | Agent learns from feedback based on actions               | Maximize total cumulative reward over time       |

---

## Summary

- **Features** are the inputs, and the **label** is the predicted output.
- In **supervised learning**, models are trained using both **features** and **labels**.
- In **unsupervised learning**, models are trained only with **features** and are used to group or reduce the data's dimensions.
- **Regression models** predict continuous values, while **classification models** predict categorical outcomes.

---

## Interview Prep: Common Questions & Concepts

### 1. What is the difference between features and labels?
   - **Features** describe the data, while **labels** are what we predict.

### 2. What is the main difference between supervised and unsupervised learning?
   - **Supervised Learning** uses **labels**, while **unsupervised learning** does not.

### 3. What type of machine learning would be used for classifying emails as spam or not?
   - **Supervised Learning (Classification Model)**

### 4. When should we use regression vs. classification models?
   - **Regression** for numerical predictions.
   - **Classification** for category-based predictions.

