
# **DL: Key Terminology**  

## **1. Neuron**  
- A **neuron** is the fundamental unit of the brain.  
- In deep learning, neurons are the basic building blocks of **neural networks**.  
- Each neuron **receives an input, processes it, and generates an output**.  

---

## **2. Multi-Layer Perceptrons (MLP)**  
- A **single neuron** is not capable of solving complex tasks.  
- A **group of neurons** forms a **multi-layer perceptron (MLP)**.  
- An MLP consists of:  
  - **Input layer**: Receives the input.
  - **Hidden layer(s)**: Perform computations on the input data.
  - **Output layer**: Produces the final result.
- Each layer has **multiple neurons**, and all neurons are **fully connected** to the next layer.  

---

## **3. Neural Network**  
- **Neural networks** are the backbone of deep learning.  
- They consist of **multiple interconnected neurons (perceptrons)**.  
- The goal of a neural network is to **find a mapping function** that fits the data.  
- **Weights and biases** are updated during training to improve accuracy.  

---

## **4. Layers in Neural Networks**  
### **Input Layer**  
- The **first layer** that receives the input data.  

### **Hidden Layers**  
- **Processing layers** between input and output.  
- These layers extract and refine important features from data.  

### **Output Layer**  
- The **final layer** that generates the output.  
- Converts processed data into a **meaningful prediction**.  

---

## **5. Weights & Bias**  
### **Weights**  
- Each **input to a neuron is multiplied** by a weight.  
- If a neuron has multiple inputs, each input has its **own weight**.  
- **Weights determine the importance** of each input.  

- Initially, weights are assigned randomly and are updated during training.
  - For example: Input `a` with weight `W1` results in an output of `a * W1`.

### **Bias**  
- A bias value is added to the weighted sum.  
- The final calculation becomes:  
  \[
  Output = (Input \times Weight) + Bias
  \]

---

## **6. Activation Function**  
- Converts the **weighted sum** of inputs into an output signal.  
- **Applies non-linearity** to the model, enabling it to learn complex patterns.  
- General form:  
  \[
  f(WX + B)
  \]  

### **Types of Activation Functions**  
- **Sigmoid**  
- **Linear**
- **ReLU (Rectified Linear Unit)**  
- **Tanh (Hyperbolic Tangent)**  
- **Softmax**  

---

## **7. Forward & Backpropagation**  
### **Forward Propagation**  
- Data **flows forward** from the input layer to the output layer.  
- Each neuron processes inputs and passes results to the next layer.  

### **Backpropagation**  
- **Error correction mechanism** used during training.  
- Steps:  
  1. Network output is compared with the actual output using a **loss function**.  
  2. Error is calculated.  
  3. Weights are adjusted **backward** to minimize the error.  

---

## **8. Cost Function**  
- Measures **how well the model's predictions match actual results**.  
- The goal of training is to **minimize the cost function**.  
- Helps improve **accuracy** and **reduce errors**.  

---

## **9. Optimizers**  
- **Optimizers adjust weights and biases** to minimize the loss function.  
- They help neural networks **learn efficiently**.  
- Examples:  
  - **Gradient Descent**  
  - **Stochastic Gradient Descent (SGD)**  

---

## **10. Gradient Descent**  
- An **optimization algorithm** used to find the minimum of the cost function.  
- Starts from a random point and **moves downhill** along the gradient.  

### **Stochastic Gradient Descent (SGD)**  
- Instead of using the entire dataset, **SGD updates weights using small batches of data**.  

---

## **11. Learning Rate**  
- **Controls how much weights are updated** in each step.  
- Needs to be chosen carefully:  
  - **Too high** → May miss the optimal point.  
  - **Too low** → Training will be slow.  

---

## **12. Batches & Epochs**  
### **Batches**  
- Instead of processing all data at once, it is split into **smaller groups (batches)**.  
- Helps improve efficiency and prevent memory issues.  

### **Epochs**  
- **One complete cycle of training** on the entire dataset.  
- More epochs = better learning (but too many may cause overfitting).  


