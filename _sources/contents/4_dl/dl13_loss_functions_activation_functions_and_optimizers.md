# DL: Loss Functions, Activation Functions, and Optimizers

---

## **1. Loss Functions**
A **loss function** measures how well a neural network models training data by comparing the target and predicted outputs. The objective during training is to minimize the loss.

### **How to Use a Loss Function in Keras**
When calling the `compile()` method, specify the loss function using the `loss` keyword argument:
```python
model.compile(loss='binary_crossentropy', optimizer=..., metrics=...)
```

---

## **Types of Loss Functions**
### **1.1 Regression Loss Functions**
Used for continuous output predictions.

- **Mean Squared Error (MSE)**: Calculates the average of the squared differences between predicted and actual values.
  ```python
  model.compile(loss='mean_squared_error', optimizer=..., metrics=...)
  ```
- **Mean Squared Logarithmic Error (MSLE)**: Computes the logarithm of predicted values before calculating MSE.
  ```python
  model.compile(loss='mean_squared_logarithmic_error', optimizer=..., metrics=...)
  ```
- **Mean Absolute Error (MAE)**: Computes the average absolute difference between actual and predicted values.
  ```python
  model.compile(loss='mean_absolute_error', optimizer=..., metrics=...)
  ```

### **1.2 Binary Classification Loss Functions**
Used when there are two output classes.

- **Binary Cross-Entropy**: Measures the difference between actual and predicted probability distributions.
  ```python
  model.compile(loss='binary_crossentropy', optimizer=..., metrics=...)
  ```
- **Hinge Loss**: An alternative to cross-entropy used in SVMs. Target values are in {-1, 1}.
  ```python
  model.compile(loss='hinge', optimizer=..., metrics=...)
  ```
- **Squared Hinge Loss**: Computes the squared version of hinge loss.
  ```python
  model.compile(loss='squared_hinge', optimizer=..., metrics=...)
  ```

### **1.3 Multi-Class Classification Loss Functions**
Used for classification problems with multiple output classes.

- **Categorical Cross-Entropy**: Measures the difference between actual and predicted probability distributions for multiple classes.
  ```python
  model.compile(loss='categorical_crossentropy', optimizer=..., metrics=...)
  ```
- **Sparse Categorical Cross-Entropy**: Similar to categorical cross-entropy but used when labels are integers instead of one-hot encoded vectors.
  ```python
  model.compile(loss='sparse_categorical_crossentropy', optimizer=..., metrics=...)
  ```

---

## **2. Activation Functions**
An **activation function** introduces non-linearity into a neural network, allowing it to learn complex patterns.

### **Types of Activation Functions**

#### **2.1 Linear Activation Function**
- A straight-line function: output = `a * input + b`.
- Outputs a constant-slope straight-line function.
- Cannot capture complex patterns.
- Example:  
     ```python
     model.add(Dense(1, activation='linear'))
     ```

#### **2.2 Sigmoid Activation Function**
- Non-linear function.
- Outputs values between 0 and 1.
- Useful for binary classification.
- Can suffer from the "vanishing gradient" problem.
- Example:  
     ```python
     model.add(Dense(1, activation='sigmoid'))
     ```

#### **2.3 Hyperbolic Tangent (Tanh) Activation Function**
- Non-linear function.
- Outputs values between -1 and 1.
- Used when data is centered around zero.
- Can also suffer from the "vanishing gradient" problem.
- Example:  
     ```python
     model.add(Dense(1, activation='tanh'))
     ```
   
#### **2.4 Rectified Linear Unit (ReLU) Activation Function**
- Non-linear function.
- Outputs positive values and zero for negative inputs.
- Helps mitigate vanishing gradients
- Can face the "dying ReLU" problem, where neurons stop updating weights.  
   - Example:  
     ```python
     model.add(Dense(64, activation='relu'))
     ```

#### **2.5 Leaky ReLU Activation Function**
- Allows a small slope for negative inputs to mitigate the dying ReLU problem.
- Example:  
     ```python
     model.add(Dense(64, activation='leaky_relu'))
     ```

### **Choosing the Right Activation Function**
- **Regression problems** → Use a linear activation function.
- **Binary classification problems** → Use a sigmoid activation function.
- **Multi-class classification problems** → Use a softmax activation function.

---

## **3. Optimizers**
An **optimizer** adjusts the neural network’s weights and learning rate to minimize the loss function.

### **Types of Optimizers**
#### **3.1 Gradient Descent**
- Updates parameters (weights) after processing all training data.
- Can be slow for large datasets.
- Example:
   ```python
   model.compile(optimizer='sgd', loss='...', metrics=['accuracy'])
   ```
   
#### **3.2 Stochastic Gradient Descent (SGD)**
- Updates parameters after each data point, rather than the entire dataset.
- Faster but introduces more variance in updates.
- Example:
   ```python
   model.compile(optimizer='sgd', loss='...', metrics=['accuracy'])
   ```

#### **3.3 Mini-Batch Gradient Descent**
- Updates parameters after processing a subset (mini-batch) of data.
- A balance between efficiency and stability.
- Example:
   ```python
   model.compile(optimizer='sgd', loss='...', metrics=['accuracy'])
   ```

---

### Summary

- **Loss functions**: Determine how well the model performs and should be minimized during training.
- **Activation functions**: Introduce non-linearity into the network, enabling it to learn complex patterns.
- **Optimizers**: Algorithms used to adjust the model parameters to minimize the loss.

By understanding these components, you can choose the most suitable options for your neural network model depending on the type of problem you're solving (regression, binary classification, multi-class classification).


