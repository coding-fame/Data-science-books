
# **DL: Multi-Layer Perceptron (MLP) Models in Keras**  

## **1. Steps to Create an MLP Model in Keras**  
1. Define **Neural Network Model**  
2. Specify **Model Inputs**  
3. Add **Model Layers**  
4. **Compile the Model**  
5. **Train the Model**  
6. **Make Predictions**  
7. **Summarize the Model**  

---

## **2. Neural Network Model in Keras**  
- `Sequential` is a **predefined class** in the Keras package.  
- It allows **easy model creation** and **layer addition**.  

```python
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(...)  # Add layers here
model.add(...)
model.add(...)
```

---

## **3. Model Inputs**  
- The **first layer** in a model must define the **input shape**.  
- This specifies the **number of input attributes** using `input_shape`.
- Example: Defining **8 input features** for a **Dense** layer.  

```python
from tensorflow.keras.layers import Dense

layer = Dense(16, input_shape=(8,))
```

---

## **4. Model Layers**  
### **Important Properties in Layers**  
Each layer has two main properties:  
1. **Weight Initialization**  
2. **Activation Function**  

---

### **1 Weight Initialization**  
- The `kernel_initializer` argument specifies the **weight initialization strategy**.  
- Common initialization methods:  
  - `random_uniform`: Small random values between **−0.05 and +0.05**.  
  - `random_normal`: Small **Gaussian random values** (mean=0, std=0.05).  
  - `zero`: **All weights initialized to zero**.  

```python
layer = Dense(16, input_shape=(8,), kernel_initializer="random_uniform")
```

---

### **2 Activation Functions**  
- Activation functions introduce **non-linearity** into the model.  
- Common activation functions:  
  - **ReLU (Rectified Linear Unit)**
  - **Sigmoid**
  - **Tanh**
  - **Softmax**  

```python
layer = Dense(16, input_shape=(8,), kernel_initializer="random_uniform", activation="relu")
```

---

### **Layer Types**  
| **Layer Type**  | **Description**  |  
|---------------|----------------|  
| **Dense**  | Fully connected layer (most commonly used in MLPs).  |  
| **Dropout**  | Reduces overfitting by randomly dropping some neurons.  |  
| **Concatenate**  | Merges outputs from multiple layers.  |  

---

## **5. Model Compilation**  
- **Compiling the model** converts it into a computational graph for training.  
- Use the `compile()` method.  
- Key arguments:  
  1. **Optimizer** (updates model weights)  
  2. **Loss Function** (measures model performance)  
  3. **Metrics** (evaluates training progress)  

```python
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

---

### **1 Model Optimizers**  
- The optimizer updates the model's weights based on the loss function.

| **Optimizer**  | **Description**  |  
|-------------|----------------|  
| **SGD**  | Stochastic Gradient Descent (supports momentum).  |  
| **RMSprop**  | Adaptive learning rate optimization.  |  
| **Adam**  | Adaptive Moment Estimation (most commonly used).  |  

```python
from tensorflow.keras.optimizers import SGD

sgd = SGD()
model.compile(optimizer=sgd)
```

---

### **2 Model Loss Functions**  
- Loss functions evaluate **model performance**.  
- Some common loss functions:  
  - **`mean_squared_error`**: Regression tasks.  
  - **`binary_crossentropy`**: Binary classification.  
  - **`categorical_crossentropy`**: Multi-class classification.  

---

### **3 Model Metrics**  
- **Metrics help track model performance** (e.g., accuracy).  

---

## **6. Model Training**  
- Train the model using the `fit()` method.  

```python
model.fit(X, y, epochs=10, batch_size=32)
```

### **1 Training Parameters**  
- **Epochs**: Number of times the model sees the full dataset.  
- **Batch Size**: Number of samples processed before updating weights.  

---

## **7. Model Prediction**  
- **Use trained model for predictions**.  
- `evaluate()` → Calculates loss and metrics for new data.  
- `predict()` → Generates output for new data.  

```python
loss, accuracy = model.evaluate(X_test, y_test)
predictions = model.predict(X_new)
```

---

## **8. Model Summary**  
- **View model architecture and layers** using `summary()`.  

```python
model.summary()
```

- **Get model configuration** using `get_config()`.  

```python
config = model.get_config()
```

---

## **9. Visualizing the Model**  
- Use `plot_model()` to generate a **visual diagram** of the model.  

```python
from tensorflow.keras.utils import plot_model

plot_model(model, to_file="model.png", show_shapes=True, show_dtype=True, 
           show_layer_names=True, expand_nested=True, show_layer_activations=True)
```

---

## **10. Summary of Steps**  
✅ **Step 1**: Create a model (`Sequential()`).  
✅ **Step 2**: Add layers (**Weights & Activation**).  
✅ **Step 3**: Compile the model (**Optimizer, Loss, Metrics**).  
✅ **Step 4**: Train the model (**Epochs & Batch Size**).  
✅ **Step 5**: Make predictions (**evaluate() & predict()**).  
✅ **Step 6**: Summarize the model (**summary() & get_config()**).  

---
