
---

# **Visualizing Model Training History in Deep Learning**  

## **1. Why Visualize Training History?**  
When training deep learning models, it's essential to **track performance** over time. **Visualization helps:**  
✅ Understand model improvement over epochs.  
✅ Detect overfitting or underfitting.  
✅ Compare training vs. validation performance.  

We will visualize:  
📈 **Accuracy Curve** – Model accuracy over training epochs.  
📉 **Loss Curve** – Model loss over training epochs.  

---

## **2. Import Required Libraries**  
```python
# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
```

---

# Steps to Visualize Model Training History

## **3. Load and Prepare Dataset**  
```python
# Load Pima Indians Diabetes dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# Split into input (X) and output (Y) variables
X = dataset[:, 0:8]  # First 8 columns as input
Y = dataset[:, 8]    # Last column as output
```

---

## **4. Build and Compile the Neural Network**  
```python
# Create a Sequential model
model = Sequential()

# Add layers
model.add(Dense(12, input_shape=(8,), activation='relu'))  # First hidden layer
model.add(Dense(8, activation='relu'))  # Second hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## **5. Train the Model & Store Training History**  
```python
# Train the model and store the history
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

# Print available history keys
print(history.history.keys())
```

✅ The `history` object stores:  
- **Training Accuracy & Loss** → `history.history['accuracy']`, `history.history['loss']`  
- **Validation Accuracy & Loss** → `history.history['val_accuracy']`, `history.history['val_loss']`  

---

## **6. Plot Training Accuracy & Validation Accuracy**  
```python
# Plot accuracy history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

📊 **Interpretation of Accuracy Plot:**  
- If **training accuracy** is high but **validation accuracy** is low → Overfitting.  
- If both **training and validation accuracy** improve → Model is learning well.  

---

## **7. Plot Training Loss & Validation Loss**  
```python
# Plot loss history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

📉 **Interpretation of Loss Plot:**  
- If **training loss decreases** but **validation loss increases** → Model is overfitting.  
- If both **loss values decrease** smoothly → Model is learning properly.  

---

## **8. Summary of Key Takeaways**  

| **Metric** | **Training Curve** | **Validation Curve** | **Expected Behavior** |
|------------|------------------|-------------------|------------------|
| **Accuracy** | Increases | Should also increase | If validation accuracy lags, overfitting is occurring. |
| **Loss** | Decreases | Should also decrease | If validation loss increases, overfitting is occurring. |

---

## Key Insights

- The **accuracy plot** helps in visualizing how well the model performs on both the training and validation datasets.
- The **loss plot** helps in understanding how quickly the model is minimizing the loss function.
- These plots can help you identify:
  - **Overfitting**: If the model performs well on the training set but poorly on the validation set, it might be overfitting.
  - **Underfitting**: If both training and validation accuracy are low, the model might be underfitting.
  - **Good Convergence**: If the accuracy increases steadily and the loss decreases, the model is likely converging well.

---

## **9. Next Steps**  
- **Monitor More Metrics**: Track precision, recall, F1-score using `tf.keras.metrics`.  
- **Use Early Stopping**: Stop training automatically when validation loss stops improving.  
- **Tune Hyperparameters**: Adjust learning rate, batch size, and epochs for better results.  

---

💡 **With these visualizations, you can track model performance and ensure optimal training!** 🚀

---