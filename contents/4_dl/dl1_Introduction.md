# DL: Introduction

## Artificial Intelligence vs Machine Learning vs Deep Learning

### **Machine Learning (ML)**
- Machine learning is a subset of artificial intelligence (AI).
- It enables systems to learn from data and make predictions on new data.

### **Deep Learning (DL)**
- Deep learning is a subset of machine learning.
- Unlike traditional machine learning, deep learning models learn features automatically from raw data.
- Machine learning models require guidance when incorrect predictions occur, while deep learning models improve autonomously.

### **Example**
- An **automatic car driving system** is an example of deep learning in action.

### AI vs ML vs DL
| Aspect              | AI                      | Machine Learning       | Deep Learning          |
|---------------------|-------------------------|------------------------|------------------------|
| **Scope**           | Broadest concept        | Subset of AI           | Subset of ML           |
| **Data Needs**      | Rule-based              | Medium datasets        | Massive datasets       |
| **Human Guidance**  | Full programming        | Feature engineering    | Automatic feature learning |
| **Example**         | Chess-playing program   | Spam filter            | Self-driving cars      |

---

## Why Deep Learning?
- Deep learning can **extract features** from raw data without manual intervention.
- It improves model accuracy as data volume increases.

### **Comparison of ML and DL Models**
In ML, models like:
- **Linear Regression**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Trees**, etc., are used for predictions.

In DL, models use **Neural Networks**, which mimic the human brain for feature extraction and problem-solving.

---

## Understanding Deep Learning
- Deep learning utilizes **Artificial Neural Networks (ANNs)**.
- It closely mimics human learning and improves automatically with data.
- Deep learning models are often referred to as **"Deep Neural Networks"**.

### **Key Features of Deep Learning**
1. **Handles Large Data Efficiently** - The more data you provide, the better the model performs.
2. **Uses Specialized Hardware** - Deep learning requires GPUs (Graphical Processing Units) and TPUs (Tensor Processing Units) for high-speed processing.

---

## Differences Between Deep Learning and Machine Learning

| Feature          | Machine Learning (ML) | Deep Learning (DL) |
|-----------------|----------------------|----------------------|
| **Data**       | Works with structured, small to medium datasets | Requires large amounts of unstructured data |
| **Hardware**   | Uses CPU (Central Processing Unit) | Uses GPU (Graphical Processing Unit) and TPU (Tensor Processing Unit) |
| **Processing** | Requires manual feature selection | Automatically learns feature selection |
| **Performance**| Good, but limited by manual tuning | Generally better due to self-optimizing algorithms |

### **Hardware Differences**
#### **CPU (Central Processing Unit)**
- Executes serial instructions.
- Good speed but consumes more memory.
- Companies: **Intel, ARM, AMD**.

#### **GPU (Graphical Processing Unit)**
- Executes parallel instructions.
- Faster than CPU with lower memory consumption.
- Companies: **NVIDIA, AMD**.

---

## Feature Selection in Machine Learning vs. Deep Learning
- In **Machine Learning**, **feature selection** is manually performed.
- In **Deep Learning**, neural networks automatically determine important features, assigning lower weights to less important ones.

---

## **Neural Networks**
### **What Are Neural Networks?**
- A network of interconnected neurons designed to solve complex problems.
- Inspired by biological neurons (brain cells).
- May contain **multiple hidden layers**.

### **Types of Neural Networks in Deep Learning**
1. **Artificial Neural Networks (ANNs)** - Also called **Feed-Forward Neural Networks**.
   - Basic building block
   - Fully connected layers
   - Used for tabular data

2. **Convolutional Neural Networks (CNNs)** - Used primarily for image processing.
   - Convolutional layers
   - Pooling operations
   - Ideal for image processing

3. **Recurrent Neural Networks (RNNs)** - Used for sequential data like speech or text.
   - Memory cells
   - Time-step processing
   - Perfect for NLP, time series

---

## 5. Why Deep Learning Wins? 🏆
- **Automatic Feature Engineering** 🛠️  
  Neural networks self-optimize feature importance through backpropagation

- **Performance Scaling** 📈  
  Accuracy improves with more data and compute power

- **Complex Pattern Recognition** 🔍  
  Handles non-linear relationships better than traditional ML

---

## Applications of Deep Learning

Deep learning is widely used across various industries, including:
- 🚗 **Self-Driving Cars**
- 📰 **Fake News Detection**
- 🗣 **Natural Language Processing (NLP)**
- 🏠 **Virtual Assistants (Alexa, Google Assistant, Siri)**
- 🎬 **Entertainment (Netflix, YouTube recommendations)**
- 📷 **Visual Recognition (Facial recognition, Object detection)**
- 💳 **Fraud Detection (Credit card fraud analysis)**
- 🏥 **Healthcare (Disease prediction, Medical imaging)**
- 🌍 **Language Translation (Google Translate, AI-driven translations)**

---

## Summary

| Aspect                        | Machine Learning                     | Deep Learning                        |
|-------------------------------|---------------------------------------|--------------------------------------|
| **Subset of**                  | AI                                    | Machine Learning                     |
| **Data Requirements**          | Works with small datasets             | Works best with large datasets       |
| **Model Guidance**             | Requires manual tuning and feature engineering | Learns automatically from raw data   |
| **Hardware**                   | Uses CPUs                             | Uses GPUs and TPUs                   |
| **Feature Selection**          | Manual intervention                   | Automatic via neural networks        |

---

## Key Takeaways 📌
- DL excels with unstructured data (images, text, audio)
- Requires significant computational resources
- Reduces need for manual feature engineering
- Continuously evolves with new architectures
- Dominates state-of-the-art AI applications

---