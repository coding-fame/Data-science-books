
---

# **Multiclass Classification: Handwritten Digits Recognition using Neural Networks**  

## **1. Overview**  
Handwritten digit classification is a classic problem in deep learning where a model predicts digits (0-9) from images. We will use a **Neural Network (NN)** to solve this problem, building upon our previous work in **logistic regression** for machine learning.  

### **1.1 Steps to Solve the Problem**  
1. **Understanding the Problem Statement**  
2. **Understanding Image Representation as Input**  
3. **Preprocessing the Data**  
4. **Building a Neural Network Model**  
5. **Training and Evaluating the Model**  
6. **Making Predictions**  

---

## **2. Understanding the Problem Statement**  
- The goal is to classify **handwritten digits (0-9)** from the **MNIST dataset** using a neural network.  
- Each digit image is **28x28 pixels**, represented as a **grayscale matrix (0-255 pixel values)**.  
- Our model will **flatten** the image into a **1D array (784 pixels)** and use a **simple neural network** to classify the digits.  

---

## **3. Understanding Image Representation as Input**  

### **3.1 How is an Image Represented in a Neural Network?**  
- An **image is a 2D matrix** where each pixel is represented by an intensity value **(0 to 255)**.  
- Before feeding it into a neural network, we **flatten** the image into a **1D array (28×28 = 784 features)**.  

### **3.2 Why Flatten the Image?**  
- Neural networks work with **1D vectors** rather than 2D images.  
- Flattening helps convert the image into a format suitable for processing by the neural network.  

---

## **4. Preprocessing the Data**  

### **4.1 Load and Split the Dataset**  
```python
# Import necessary libraries
from tensorflow import keras
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Display dataset dimensions
print(f"Training data shape: {X_train.shape}")  # (60000, 28, 28)
print(f"Testing data shape: {X_test.shape}")    # (10000, 28, 28)
```

### **4.2 Visualizing Sample Images**  
```python
# Show the first image in training set
plt.matshow(X_train[0], cmap='gray')
plt.show()

# Display corresponding label
print(f"Label: {y_train[0]}")
```

### **4.3 Flattening the Images (Convert 2D to 1D)**  
```python
# Flatten the images from 28x28 to a 1D array of 784 pixels
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

# Print new shape
print(f"Flattened training data shape: {X_train_flattened.shape}")  # (60000, 784)
print(f"Flattened testing data shape: {X_test_flattened.shape}")    # (10000, 784)
```

### **4.4 Feature Scaling**  
```python
# Normalize pixel values to range [0, 1] for better model performance
X_train = X_train / 255.0
X_test = X_test / 255.0
```

---

## **5. Building the Neural Network Model**  

```python
# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential()

# Input layer: 784 neurons (flattened 28x28 image)
# Output layer: 10 neurons (one for each digit 0-9) with softmax activation
model.add(Dense(10, input_shape=(784,), activation='sigmoid'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display model summary
model.summary()
```

### **Why Use `sparse_categorical_crossentropy`?**  
- Since our labels (y_train, y_test) are integers (0-9), we use `sparse_categorical_crossentropy` instead of `categorical_crossentropy`, which is used for one-hot encoded labels.  

---

## **6. Training and Evaluating the Model**  

### **6.1 Train the Model**  
```python
# Train the neural network
model.fit(X_train_flattened, y_train, epochs=5, batch_size=32)
```

### **6.2 Evaluate Model Performance**  
```python
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test_flattened, y_test)

print(f"Test Accuracy: {test_acc*100:.2f}%")
```

---

## **7. Making Predictions**  

### **7.1 Predict on Test Data**  
```python
# Predict the first test image
y_predicted = model.predict(X_test_flattened)

# Print predicted probabilities for each class (0-9)
print(y_predicted[0])
```

### **7.2 Visualizing Predictions**  
```python
# Display the first test image
plt.matshow(X_test[0], cmap='gray')
plt.show()

# Display the predicted label
predicted_label = y_predicted[0].argmax()
print(f"Predicted Label: {predicted_label}")
```

```python
# Display another test image with prediction
plt.matshow(X_test[1], cmap='gray')
plt.show()

predicted_label = y_predicted[1].argmax()
print(f"Predicted Label: {predicted_label}")
```

---

## **8. Summary of Key Takeaways**  

| **Step** | **Description** |
|----------|---------------|
| **Dataset** | We used the **MNIST dataset** (60,000 training, 10,000 testing images). |
| **Preprocessing** | Images were **flattened** from **28×28 to 784 features** and **normalized** to [0,1]. |
| **Neural Network** | A **simple 2-layer NN** with 10 output neurons and sigmoid activation. |
| **Loss Function** | Used `sparse_categorical_crossentropy` for integer-labeled output. |
| **Evaluation** | Achieved **high accuracy** on test data. |
| **Predictions** | Model successfully predicts digits from unseen test images. |

---

## **9. Final Thoughts**  

- **This is a simple neural network**, but adding **hidden layers and using ReLU activation** can improve accuracy.  
- **Try experimenting** with more layers, different activation functions, and optimizers for better performance.  
- **Next Steps**: Implement **Convolutional Neural Networks (CNNs)** for even better results in handwritten digit classification!  

