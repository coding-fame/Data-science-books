
---

# **Checkpointing in Deep Learning: Save the Best Model**  

## **1. Why Use Checkpointing?**  
Training deep learning models can take **hours, days, or even weeks**. If training stops unexpectedly (e.g., power failure, system crash), you may lose all progress. **Checkpointing** helps by:  
✅ Saving model weights at regular intervals.  
✅ Resuming training without starting from scratch.  
✅ Keeping the **best-performing** model instead of the last saved version.  

---

## **2. What Does a Checkpoint Save?**  
- **Model Weights** → Stored in **HDF5 (`.hdf5`)** format.  
- **Best Model Selection** → Saves only the best model based on a chosen metric (e.g., validation accuracy).  

---

## **3. Implementing Checkpointing in Keras**  

### **3.1 Import Required Libraries**  
```python
# Import necessary libraries
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.callbacks import ModelCheckpoint
```

### **3.2 Load and Prepare the Dataset**  
```python
# Load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (y) variables
X = dataset[:, 0:8]  # First 8 columns as input
y = dataset[:, 8]    # Last column as output
```

### **3.3 Define the Deep Learning Model**  
```python
# Define a simple neural network
model = Sequential()

model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## **4. Setting Up Model Checkpointing**  

### **4.1 Define Checkpoint Parameters**  
```python
# Filepath format for saving model weights
filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

# Define checkpoint callback
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                             verbose=1, save_best_only=True, mode='max')

# Create a list of callbacks
callbacks_list = [checkpoint]
```

### **4.2 Train the Model with Checkpointing**  
```python
# Train the model with checkpoint callback
model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10, 
          callbacks=callbacks_list, verbose=0)
```

✅ **Now, during training, the model saves the best version based on validation accuracy!**  

---

## **5. How the Checkpoint Works**  
- `filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"`  
  - Saves weights with epoch number and validation accuracy.  
  - Example: `weights-improvement-10-0.87.hdf5` (Saved at epoch 10, val_accuracy = 87%).  
- `monitor='val_accuracy'`  
  - Monitors **validation accuracy** to save the best-performing model.  
- `save_best_only=True`  
  - Saves only if the model improves (prevents unnecessary file storage).  
- `mode='max'`  
  - Ensures the model saves when validation accuracy **increases**.  

---

## **6. Summary of Key Takeaways**  

| **Feature** | **Description** |
|------------|---------------|
| **Why Use Checkpoints?** | Saves model progress to prevent loss of training time. |
| **What Gets Saved?** | Model **weights** in **HDF5 format** (`.hdf5`). |
| **How is the Best Model Chosen?** | Based on highest **validation accuracy (`val_accuracy`)**. |
| **File Naming Format** | Includes **epoch number** and **validation accuracy**. |
| **How to Resume Training?** | Load the best saved weights and continue training. |

---

## **7. Next Steps**  
- **Load the Best Model**: Use `model.load_weights()` to resume training from the best checkpoint.  
- **Fine-tune Checkpoint Strategy**: Experiment with different checkpointing criteria (e.g., monitoring loss instead of accuracy).  
- **Save the Entire Model**: Explore `model.save()` to store both architecture and weights.  

---

💡 **With checkpointing, you ensure no training progress is lost and always keep the best-performing model!** 🚀

---