# DL: Neuron

## **Understanding a Neuron**

- A neuron, similar to those in the human brain, forms the basic structure of a neural network.  
- When we receive information, we process it and generate an output.  
- Similarly, in a neural network, a neuron receives an input, processes it, and generates an output.  
- This output can be the final result or can be passed to other neurons for further processing.  

---

## **Use Case: Predicting Insurance Purchase Based on Age**

### **Scenario**
- We have a dataset where we need to predict if a person will buy insurance based on their age.
- In our previous exploration of Machine Learning, we solved this using **Logistic Regression**.

### **Dataset Information**
- **0** → Person did not buy insurance.
- **1** → Person bought insurance.
- Young people are less likely to buy insurance.
- As age increases, the probability of purchasing insurance also increases.

### **Problem Statement**
- Given the **age** of a person, predict whether they will buy insurance or not.

---

## **Data Visualization**
- A **scatter plot** can be used to visualize the given dataset.
- Observing the data points helps in identifying patterns.

### **What Line Fits This Kind of Data?**
- A separation line can be drawn, which helps in making predictions for new data points.
- This line helps in **classification**, making it easy to distinguish between those who buy insurance and those who do not.

### **What is this Line?**
- This line represents the **Sigmoid Function** (also called the **Logit Function**).

---

## **Sigmoid Function (Logit Function)**

- The **Sigmoid function** is a mathematical function used to map predicted values to probabilities.
- It transforms any real value into a range between **0 and 1**.
- The output forms an "S"-shaped curve, which helps in **classification tasks**.

### **Formula:**

\[ \sigma(y) = \frac{1}{1 + e^{-y}} \]

where:
- **y** is the linear equation output.
- **e** is Euler’s number (approximately 2.718).

### **Equation Parameters in Logistic Regression**
- Values are obtained from the trained **logistic regression model**:
  - **m (slope)** = **0.0042**
  - **b (bias)** = **-1.53**

---

## **Example Calculations**

### **Prediction for Age = 35**
1. Plug age = 35 into the logistic regression equation: `y = mx + b`.
2. The result is `y = 0.0042 * 35 - 1.53 = -0.70`.
3. Apply **Sigmoid function** to get the probability. `sigmoid(-0.70) = 0.48`.
4. The final result is **0.48**, which is **< 0.5** → Person **will not** buy insurance (**Red color**).

### **Prediction for Age = 43**
1. Plug age = 43 into the logistic regression equation: `y = mx + b`.
2. The result is `y = 0.0042 * 43 - 1.53 = -0.37`.
3. Apply the sigmoid function: `sigmoid(-0.37) = 0.57`. 
4. The final result is **0.57**, which is **> 0.5** → Person **will** buy insurance (**Green color**).

---

## **Neuron Representation**

### **Understanding the Neuron Process**
- The above process mimics how a **neuron** functions.
- A neuron consists of:
  1. **Linear Equation** (First part → `y = mx + b`)
  2. **Activation Function** (Second part → Sigmoid function)
  
### **Using Multiple Features**
- Instead of using **only age**, we can consider multiple features.
- **Neuron Representation:**
  - **X1, X2, X3** → Input features.
  - **W1, W2, W3** → Weights assigned to features.
  - **b** → Bias term.
  
\[ y = W1X1 + W2X2 + W3X3 + b \]

- The neuron **automatically adjusts the weights** to learn from data.

- The neuron computes a weighted sum of the input features and passes it through the activation function.

---

## **Conclusion**
- **Neurons** form the building blocks of **Deep Learning models**.
- **Activation functions**, like **sigmoid**, help in making predictions.
- **Multiple features** can be used to improve predictions.
- Understanding how neurons work is key to mastering **Neural Networks**.

---

