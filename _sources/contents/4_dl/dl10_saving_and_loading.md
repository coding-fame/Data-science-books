# **Saving and Loading a Deep Learning Model in Keras**  

## **1. Overview**  
When working with deep learning models, we often need to **save** and **reload** them for later use. Saving a trained model allows us to:  
- Avoid retraining from scratch.  
- Deploy the model in a production environment.  
- Share the model with others.  

### **1.1 How is a Model Saved?**  
In **Keras**, we typically save:  
1. **Model architecture** → Stored in a **JSON** file.  
2. **Model weights** → Stored in an **HDF5 (Hierarchical Data Format)** file.  

---

## **2. Why Use HDF5 Format?**  
- **Optimized for large numerical datasets** (like deep learning model weights).  
- **Faster loading times** compared to other formats.  
- **Portable and scalable**, making it ideal for storing trained models.  

---

## **3. Saving a Deep Learning Model**  

### **3.1 Import Required Libraries**  
```python
# Import necessary libraries
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.models import model_from_json
```

### **3.2 Load the Dataset**  
```python
# Load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y) variables
X = dataset[:, 0:8]  # First 8 columns as input
y = dataset[:, 8]    # Last column as output
```

### **3.3 Define and Train the Model**  
```python
# Define a simple neural network model
model = Sequential()

model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=150, batch_size=10)
```

### **3.4 Save the Model Architecture (JSON Format)**  
```python
# Convert the model to JSON format
model_json = model.to_json()

# Save the JSON file
with open("model.json", "w") as json_file:
    json_file.write(model_json)

print("Model architecture saved as model.json")
```

### **3.5 Save the Model Weights (HDF5 Format)**  
```python
# Save model weights in HDF5 format
model.save_weights("model.h5")

print("Model weights saved as model.h5")
```

✅ **At this point, we have successfully saved our model!** 🎉  

---

## **4. Loading a Saved Model**  

### **4.1 Import Required Libraries**  
```python
# Import necessary libraries
from keras.models import model_from_json
import numpy as np
```

### **4.2 Load the Dataset Again**  
```python
# Load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]
```

### **4.3 Load the Model from JSON File**  
```python
# Open the JSON file
json_file = open('model.json', 'r')

# Read the JSON file
model_j = json_file.read()
json_file.close()

# Load the model architecture from JSON
model = model_from_json(model_j)

print("Model architecture loaded from model.json")
```

### **4.4 Load the Model Weights from HDF5 File**  
```python
# Load model weights
model.load_weights("model.h5")

print("Model weights loaded from model.h5")
```

### **4.5 Compile and Evaluate the Model**  
```python
# Compile the loaded model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Evaluate the model on the dataset
score = model.evaluate(X, y)

print(f"Loaded Model Accuracy: {score[1]*100:.2f}%")
```

---

## **5. Summary of Key Takeaways**  

| **Step** | **Description** |
|----------|---------------|
| **Save Model Architecture** | Stored as **JSON file (`model.json`)** |
| **Save Model Weights** | Stored as **HDF5 file (`model.h5`)** |
| **Load Model Architecture** | Read from **JSON file** |
| **Load Model Weights** | Read from **HDF5 file** |
| **Compile & Evaluate Model** | Test model performance on the dataset |

---

## **6. Final Thoughts**  

- Saving and loading models is **essential** in deep learning workflows.  
- HDF5 format is **efficient** for storing large model weights.  
- **Next Steps**: Explore **saving and loading entire models (`model.save()`)**, including optimizer states!  

