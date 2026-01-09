# MLflow: Interview Guide

## **What is MLflow?**

MLflow is a free tool that helps manage the entire machine learning (ML) process. It makes it easier to:

- **Track Experiments**: Save and compare model settings and results.
- **Reproduce Results**: Ensure models can be recreated.
- **Deploy Models**: Easily put models into production.
- **Manage Models**: Keep track of different versions of models.

MLflow works with popular ML libraries like **TensorFlow, PyTorch, and Scikit-learn**.

---

## **Main Parts of MLflow**

### **1. MLflow Tracking**
- Records model performance and settings.
- Helps compare different experiments.
- Simple to use with a few lines of code.

### **2. MLflow Projects**
- Standardizes ML code for consistency.
- Uses `requirements.txt` or `conda.yaml` to manage dependencies.

### **3. MLflow Models**
- Saves models in a universal format.
- Makes it easy to deploy models anywhere.

### **4. MLflow Model Registry**
- Stores and tracks different versions of models.
- Helps teams collaborate on model development.

---

## **Why Use MLflow?**

| Feature            | Benefit |
|-------------------|----------|
| **Experiment Tracking** | Saves model settings and results. |
| **Reproducibility** | Ensures models can be recreated exactly. |
| **Model Management** | Keeps track of different versions. |
| **Deployment** | Easily deploy models anywhere. |
| **Collaboration** | Helps teams work together efficiently. |
| **Scalability** | Works with cloud platforms and large projects. |

---

## **How to Use MLflow**

### **1. Install MLflow**
Install MLflow with:
```bash
pip install mlflow
```

---

### **2. Track Experiments**
Log settings and results for a model:
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Enable auto-logging
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")
    print(f"Model logged with accuracy: {accuracy}")
```

---

### **3. View Results in MLflow UI**
Start the MLflow UI:
```bash
mlflow ui
```
- Open [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in your browser.

---

### **4. Register and Deploy Models**
Register a trained model:
```bash
mlflow models serve --model-uri models:/random_forest_model/1 --port 1234
```

---

### **5. Deployment Options**
MLflow supports various deployment methods:
- **REST API**: Serve models as an API.
- **Docker/Kubernetes**: Deploy in containers.
- **Cloud Platforms**: Deploy to AWS, Azure, or Google Cloud.

---

## **Important MLflow Commands & APIs**

### **1. Experiment Tracking**

| Command | What It Does |
|--------|-------------|
| `mlflow.start_run()` | Starts a new MLflow run. |
| `mlflow.log_param("param_name", value)` | Saves a model parameter. |
| `mlflow.log_metric("metric_name", value)` | Saves a model performance metric. |
| `mlflow.log_artifact("file_path")` | Saves a file related to the experiment. |

Example:
```python
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

---

### **2. MLflow Models**

| Command | What It Does |
|--------|-------------|
| `mlflow.save_model(model, "path")` | Saves a model. |
| `mlflow.log_model(model, "artifact_path")` | Logs a model. |
| `mlflow.models.serve(model_uri, port=5000)` | Serves a model as an API. |
| `mlflow.models.predict(model_uri, data)` | Runs predictions with a saved model. |

Example:
```python
mlflow.sklearn.log_model(model, "sklearn_model")
```

---

### **3. MLflow Model Registry**

| Command | What It Does |
|--------|-------------|
| `mlflow.register_model("model_uri", "name")` | Adds a model to the registry. |
| `mlflow.transition_model_version_stage("name", version, "stage")` | Moves a model to "Staging" or "Production". |
| `mlflow.get_model_version("name", version)` | Gets details of a model version. |

Example:
```python
mlflow.register_model("runs:/<run_id>/model", "MyModel")
```

---

### **4. MLflow Auto-Logging**
Enable automatic logging for different ML frameworks:
```python
mlflow.sklearn.autolog()  # For Scikit-learn
mlflow.tensorflow.autolog()  # For TensorFlow/Keras
```

---

## **When to Use MLflow?**

1. **Trying different ideas**: When testing many model versions
2. **Team projects**: When multiple people work together
3. **Important models**: When you need to track model history
4. **Moving to real use**: When putting models in apps/websites

---

## **Conclusion**

MLflow makes ML development **organized, reproducible, and scalable**. By using MLflow, you can easily **track, manage, and deploy models**, making your ML projects more efficient and professional.


