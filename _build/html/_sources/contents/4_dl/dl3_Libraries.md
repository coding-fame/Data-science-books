# DL: Libraries

---

## TensorFlow

✅ **TensorFlow** is a deep learning framework developed by **Google** in 2015. It is open-source and widely used for numerical computation and deep learning tasks.

### Key Features:
- Written in Python and optimized for performance.
- Uses **graph computation** for faster execution.
- Enables scalable machine learning with GPU and TPU support.
- Provides high-level APIs for easy implementation.

### Installation:
```sh
pip install tensorflow
```

### Update:
- From **TensorFlow 2.0**, **Keras** is built-in, so no separate installation is required.

---

## PyTorch

✅ **PyTorch** is a deep learning framework developed by **Facebook** in 2016. It is also open-source and popular for its dynamic computational graph and ease of use.

### Key Features:
- More **Pythonic** and intuitive than TensorFlow.
- Offers **eager execution**, making debugging easier.
- Strong support for **GPU acceleration**.
- Widely used in **research and production**.

🔹 **Note:** CNTK (developed by Microsoft) is another deep learning framework, but it is not as popular as TensorFlow and PyTorch.

---

## Keras


✅ **Keras** is an open-source high-level neural network API written in **Python**. It supports **TensorFlow**, **Theano**, and **CNTK** as backends.

### Key Features:
- Developed by **Francois Chollet** (Google engineer).
- **User-friendly** and easy to use.
- **Extensible**, allowing custom layers and models.
- Supports **Convolutional** and **Recurrent Neural Networks**.

It is an **Application Programming Interface (API)** that allows developers to define and train **artificial neural networks (ANNs)**

---

## Steps to Create Deep Learning Models with Keras

---

### 1️⃣ **Define Your Model** 
- Create a **Sequential** model and add layers.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

### 2️⃣ **Compile Your Model**
- Specify the **loss function**, **optimizer**, and evaluation **metrics**.
```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

### 3️⃣ **Train Your Model**
- Use the **fit()** method to train on data.
```python
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
```
  
---

### 4️⃣ **Make Predictions** 
- Use **evaluate()** for testing and **predict()** for new data.
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

predictions = model.predict(X_new)
print(predictions)
```

---

## Why Use Keras?

📌 **Beginner-Friendly** – Simplifies deep learning implementation.  
📌 **Flexible** – Works seamlessly with different deep learning frameworks.  
📌 **Fast Development** – Ideal for quick model prototyping and experimentation.  
📌 **Scalable** – Supports both simple and complex neural networks.  

---

📌 **Summary:**
- **TensorFlow** is the most widely used deep learning library.
- **PyTorch** is flexible and popular in research.
- **Keras** provides a high-level API for building deep learning models easily.

