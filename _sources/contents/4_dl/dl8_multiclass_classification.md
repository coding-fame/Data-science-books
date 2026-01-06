# **Multiclass Classification of IRIS Species using Deep Learning**  

## **1. Overview**  
Multiclass classification is a type of supervised learning where we classify inputs into **three or more categories**.  

In this guide, we will solve the **Iris flower classification problem** using **Deep Learning** with TensorFlow/Keras.  

### **1.1 Steps to Solve the Problem**  
1. **Problem Description**  
2. **Import Required Libraries**  
3. **Load the Dataset**  
4. **Encode the Output Variable**  
5. **Evaluate the Model using k-Fold Cross-Validation**  
6. **Complete Code Example**  

---

## **2. Problem Description**  

The **Iris dataset** is a famous dataset used for classification tasks. It contains **150 samples** of **three different flower species**:  
- **Setosa**  
- **Versicolor**  
- **Virginica**  

Each sample has **four numeric features** measured in centimeters:  
1. **Sepal Length**  
2. **Sepal Width**  
3. **Petal Length**  
4. **Petal Width**  

This is a **multiclass classification problem**, meaning our model needs to classify an input into one of the three categories.  

---

## **3. Import Required Libraries**  

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from scikeras.wrappers import KerasClassifier
```

---

## **4. Load the Dataset**  

We can load the **Iris dataset** using `pandas.read_csv()`. Make sure the dataset is in the correct format.

```python
# Load dataset
dataframe = pd.read_csv("iris.csv", header=None)
dataset = dataframe.values

# Split into input (X) and output (y) variables
X = dataset[:, 0:4].astype(float)  # Features
Y = dataset[:, 4]  # Target labels (species)
```

---

## **5. Encode the Output Variable**  

Since our target variable (`Y`) consists of **categorical labels** (flower species), we need to **convert it into numerical values** using **Label Encoding** and **One-Hot Encoding**.

```python
# Encode class values as integers
encoder = LabelEncoder()
# Label encode the target variable
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Convert integers to one-hot encoding
dummy_y = to_categorical(encoded_Y)
```

---

## **6. Define and Evaluate the Model using k-Fold Cross-Validation**  

### **6.1 Define the Model**  

We will create a simple **feedforward neural network** with:  
- **4 input neurons** (one for each feature)  
- **1 hidden layer** with **8 neurons** and **ReLU activation**  
- **3 output neurons** (one for each class) with **softmax activation**  

```python
# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Define the neural network model
def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(8, input_shape=(4,), activation='relu'))  # Hidden layer
    model.add(Dense(3, activation='softmax'))  # Output layer (3 classes)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

---

### **6.2 Evaluate the Model with k-Fold Cross-Validation**  

k-Fold Cross-Validation helps in getting a **more reliable accuracy estimate** by splitting the dataset into multiple training and validation subsets.  

```python
# Wrap the model with KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

# Define k-Fold Cross-Validation
kfold = KFold(n_splits=3, shuffle=True, random_state=seed)

# Evaluate the model using cross-validation
results = cross_val_score(estimator, X, dummy_y, cv=kfold)

# Print model accuracy
print(f"Accuracy: {results.mean()*100:.2f}% (+/- {results.std()*100:.2f}%)")
```

---

## **7. Summary of Key Takeaways**  

| **Step** | **Description** |
|----------|---------------|
| **Dataset** | The Iris dataset contains **150 samples** of 3 species. |
| **Feature Encoding** | We used **Label Encoding** and **One-Hot Encoding** for categorical labels. |
| **Neural Network** | A simple **2-layer feedforward network** was created using Keras. |
| **Evaluation** | We used **k-Fold Cross-Validation** (k=3) to evaluate model performance. |
| **Loss Function** | `categorical_crossentropy` is used for multiclass classification. |
| **Activation Functions** | `ReLU` for hidden layer, `Softmax` for output layer. |
| **Optimizer** | `Adam` optimizer for efficient training. |

---

## **8. Final Thoughts**  

- Multiclass classification problems like **Iris Classification** are great for learning **deep learning basics**.  
- The **k-Fold Cross-Validation** technique helps in achieving a **more generalized** model performance.  
- You can **tweak** the number of neurons, activation functions, or optimizer to experiment with better accuracy.  
