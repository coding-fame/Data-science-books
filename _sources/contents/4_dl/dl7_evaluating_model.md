
# **DL: Evaluating Model Performance**  

## **1. Overview**  
Evaluating a deep learning model is **crucial** to understanding how well it performs on unseen data.  

### **1.1 Ways to Evaluate a Model**  
We can evaluate a model using the following techniques:  
1. **Data Splitting**  
   - **Automatic Verification Dataset** (via `validation_split`)  
   - **Manual Verification Dataset** (via `train_test_split`)  
2. **Manual k-Fold Cross-Validation**  

---

## **2. Data Splitting - Automatic Verification Dataset**  

### **2.1 What is Automatic Verification?**  
- Instead of manually creating a validation set, we use the `validation_split` argument in `model.fit()`.  
- This **automatically** holds back a portion of the data for validation.  
- Common values: **0.2 (20%)** or **0.33 (33%)**.  

### **2.2 Code Implementation**  

```python
# Import required libraries
import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Fix random seed for reproducibility
np.random.seed(7)

# Load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y)
X = dataset[:, 0:8]
y = dataset[:, 8]

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with automatic validation split
model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10)
```

### **2.3 Observations**  
- The **verbose output** during training will show **loss and accuracy** for both the **training dataset** and the **validation dataset**.  
- This helps us understand whether the model is **overfitting** or **generalizing well**.  

---

## **3. Data Splitting - Manual Verification Dataset**  

### **3.1 What is Manual Verification?**  
- We manually **split** the dataset into **training** and **testing** sets using `train_test_split()`.  
- This ensures that the model is trained on **one set** and evaluated on **another**, reducing overfitting.  

### **3.2 Code Implementation**  

```python
# Import required libraries
import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y)
X = dataset[:, 0:8]
y = dataset[:, 8]

# Split into 67% train and 33% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=10)

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

### **3.3 Observations**  
- We **explicitly** create **training (67%)** and **testing (33%)** sets.  
- The model is trained on `X_train, y_train` and tested on `X_test, y_test`.  
- `evaluate()` computes **loss** and **accuracy** on the test dataset.  

---

## **4. Manual k-Fold Cross-Validation**  

### **4.1 What is k-Fold Cross-Validation?**  
- Instead of a **single** train-test split, k-Fold **splits the dataset into k subsets** (folds).  
- The model is trained and validated **k times**, each time using a different subset as validation.  
- This helps in **reducing variance** and getting a **more reliable performance estimate**.  

### **4.2 Code Implementation**  

```python
# Import required libraries
import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y)
X = dataset[:, 0:8]
y = dataset[:, 8]

# Define 10-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvscores = []

for train, test in kfold.split(X, y):
    # Create model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X[train], y[train], epochs=150, batch_size=10, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X[test], y[test], verbose=0)
    print(f"Fold Accuracy: {scores[1] * 100:.2f}%")
    cvscores.append(scores[1] * 100)

# Print final model performance
print(f"Mean Accuracy: {np.mean(cvscores):.2f}% (+/- {np.std(cvscores):.2f}%)")
```

### **4.3 Observations**  
- The dataset is **split into 10 folds** (`n_splits=10`).  
- The model is trained and tested **10 times**, each time with a **different validation fold**.  
- Finally, we compute the **mean accuracy** across all folds.  

---

## **5. Summary of Model Evaluation Techniques**  

| Evaluation Method | Description | Pros | Cons |
|------------------|-------------|------|------|
| **Automatic Verification (`validation_split`)** | Splits the training data automatically for validation. | Quick & easy | Can lead to **data leakage** if dataset is small. |
| **Manual Verification (`train_test_split`)** | Splits dataset manually into train and test sets. | More control over the split | Less robust compared to cross-validation. |
| **k-Fold Cross-Validation** | Splits dataset into k subsets and trains model multiple times. | More reliable & reduces variance | Computationally expensive. |

---

## **6. Final Thoughts**  
- Always **evaluate** your deep learning models to ensure they generalize well to unseen data.  
- If **data is limited**, prefer **k-Fold Cross-Validation** over a single **train-test split**.  
- If working with **large datasets**, **automatic verification (`validation_split`)** is a good choice.  
