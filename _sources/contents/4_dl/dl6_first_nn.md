
# **DL: First Neural Network with Keras**  

---

## **1. Steps to Implement Your First Neural Network**  
1. **Load Dataset** (Pima Indians Diabetes)  
2. **Define the Keras Model**  
3. **Compile the Model**  
4. **Train (Fit) the Model**  
5. **Evaluate the Model**  
6. **Make Predictions**  
7. **Summarize the Model**  
8. **Visualize the Model**  

---

## **2. Dataset Explanation**  

We will use the **Pima Indians Diabetes dataset** (`pima-indians-diabetes.csv`), a well-known **healthcare dataset** for binary classification.  

### **About the Dataset**  
- The dataset consists of **medical records** of **Pima Indian women (aged 21 and older)**.  
- The task is to **predict the onset of diabetes** within **five years**.  
- The dataset contains **only numerical features**.  

### **Problem Type**  
✅ **Binary Classification** (1 = **Diabetic**, 0 = **Non-Diabetic**)  

---

## **3. Input and Output Variables**  

### **3.1 Input Features (X)**  
| #  | Feature Name | Description |
|----|-------------|-------------|
| 1  | **Pregnancies** | Number of times pregnant |
| 2  | **Glucose** | Plasma glucose concentration (2 hours) |
| 3  | **Blood Pressure** | Diastolic blood pressure (mm Hg) |
| 4  | **Skin Thickness** | Triceps skin fold thickness (mm) |
| 5  | **Insulin** | 2-hour serum insulin (μIU/ml) |
| 6  | **BMI** | Body Mass Index (kg/m²) |
| 7  | **Diabetes Pedigree Function** | Genetic risk of diabetes |
| 8  | **Age** | Patient’s age (years) |

### **3.2 Output Variable (y)**  
| Value | Meaning |
|--------|------------|
| **0**  | Non-Diabetic |
| **1**  | Diabetic |

---

## **4. Load the Dataset**  

We will use **NumPy** to load the dataset and split it into **input (X) and output (y) variables**.  

```python
from numpy import loadtxt

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y)
X = dataset[:, 0:8]
y = dataset[:, 8]
```

---

## **5. Define the Keras Model**  
- Keras models are built using **layers**.  
- We define a **Sequential** model with three layers.  

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))  # Input + First Hidden Layer
model.add(Dense(8, activation='relu'))  # Second Hidden Layer
model.add(Dense(1, activation='sigmoid'))  # Output Layer
```

---

## **6. Understanding the Model**  

### **6.1 Input Shape & Activation Functions**  
✅ **First Hidden Layer**: 12 neurons, **ReLU** activation  
✅ **Second Hidden Layer**: 8 neurons, **ReLU** activation  
✅ **Output Layer**: 1 neuron, **Sigmoid** activation (for binary classification)  

> **Why ReLU?** Helps prevent the vanishing gradient problem.  
> **Why Sigmoid?** Outputs probabilities between 0 and 1, perfect for binary classification.  

---

## **7. Compile the Model**  
Before training, we need to **compile** the model.  

```python
# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### **7.1 Key Components**  
| Component | Description |
|-----------|------------|
| **Loss Function** | `"binary_crossentropy"` (since it's a binary classification problem) |
| **Optimizer** | `"adam"` (Adaptive Moment Estimation, a powerful gradient descent algorithm) |
| **Metrics** | `accuracy` (to evaluate performance) |

---

## **8. Train (Fit) the Model**  
Now, we **train the model** on the dataset using the `fit()` method.  

```python
# Train the model
model.fit(X, y, epochs=150, batch_size=10)
```

### **8.1 Training Parameters**  
✅ **Epochs (150)** → Number of passes over the dataset  
✅ **Batch Size (10)** → Number of samples before updating weights  

---

## **9. Evaluate the Model**  
Once training is complete, we evaluate its performance using the `evaluate()` method.  

```python
# Evaluate the model
_, accuracy = model.evaluate(X, y)

# Print accuracy
print('Accuracy: %.2f' % (accuracy * 100))
```

### **9.1 What Does Evaluate Return?**  
- **Loss Value**  
- **Accuracy**  

> 🎯 Higher accuracy means a better-performing model!  

---

## **10. Make Predictions**  
Now, let's use the trained model to make predictions.  

```python
# Make class predictions
predictions = (model.predict(X) > 0.5).astype(int)

# Print predictions for first 5 samples
for i in range(5):
    print(X[i], "---", predictions[i], "---", y[i])
```

### **10.1 Why `predict(X) > 0.5`?**  
- The output layer has a **sigmoid activation function**, which outputs probabilities.  
- We classify **values ≥ 0.5 as 1** (diabetic) and **< 0.5 as 0** (non-diabetic).  

---

## **11. Optional: Train Without Progress Bar**  
Setting `verbose=0` hides the progress bar during training.  

```python
# Train the model without progress bar
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
```

---

## **12. Model Summary**  
View the model architecture using `summary()`.  

```python
# Display model summary
model.summary()
```

---

## **13. Visualizing the Model**  
Generate an **image of the model architecture** using `plot_model()`.  

```python
from tensorflow.keras.utils import plot_model 

plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, 
           show_layer_names=True, expand_nested=True, show_layer_activations=True)
```

---

## **14. Final Summary of Steps**  
✅ **Step 1**: Load the dataset (`pima-indians-diabetes.csv`)  
✅ **Step 2**: Define the model (`Sequential()`)  
✅ **Step 3**: Add layers (`Dense()`) with activation functions  
✅ **Step 4**: Compile the model (`compile()`)  
✅ **Step 5**: Train the model (`fit()`)  
✅ **Step 6**: Evaluate the model (`evaluate()`)  
✅ **Step 7**: Make predictions (`predict()`)  
✅ **Step 8**: Summarize & visualize the model (`summary()` & `plot_model()`)  

---
