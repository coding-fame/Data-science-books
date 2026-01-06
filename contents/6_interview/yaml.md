# YAML Crash Course

## 📌 What is YAML?  
YAML (**YAML Ain't Markup Language**) is a **human-readable data serialization format** commonly used for **configuration files, data exchange, and structured data representation**. It is widely used in **DevOps, Kubernetes, CI/CD pipelines, and Machine Learning workflows**.

> **Think of YAML as JSON’s simpler, more readable sibling!**

---

## 🔹 Why Use YAML?
✅ **Easy to Read** – Uses indentation instead of brackets or commas  
✅ **Lightweight** – No unnecessary symbols, just plain text  
✅ **Widely Used** – Found in **Kubernetes, Docker, CI/CD, MLflow, and API configs**  
✅ **Supports Comments** – Unlike JSON, YAML allows comments  
✅ **Cross-Language Support** – Works with Python, JavaScript, Java, etc.  

---

## 🔹 YAML Syntax Basics

### 1️⃣ **Key-Value Pairs** (Like a Dictionary)
```yaml
name: John Doe
age: 30
is_happy: true
```

### 2️⃣ **Lists (Arrays)**
```yaml
fruits:
  - Apple
  - Banana
  - Cherry
```
(Same as `fruits: ["Apple", "Banana", "Cherry"]` in JSON)

### 3️⃣ **Nested Data (Hierarchy/Indentation Matters!)**  
```yaml
person:
  name: Alice
  address:
    city: New York
    zip: 10001
```

### 4️⃣ **Multi-Line Strings (For Longer Text)**
```yaml
bio: |
  Alice is a data scientist.
  She loves machine learning and deep learning.
```

### 5️⃣ **Using Variables & Reuse (Anchors & Aliases)**
```yaml
default_config: &config
  batch_size: 32
  learning_rate: 0.001

model1:
  <<: *config  # Reuses default_config
  epochs: 10

model2:
  <<: *config
  epochs: 50
```
🔥 **ML models can share the same config and override specific values!**

---

## 🔹 YAML vs JSON vs XML (Comparison)
| Feature        | YAML 🟡 | JSON 🟢 | XML 🔵 |
|---------------|--------|--------|--------|
| **Readability** | ✅ Easy | ⚠️ Okay | ❌ Hard |
| **Size** | ✅ Small | ⚠️ Medium | ❌ Large |
| **Supports Comments?** | ✅ Yes (`# Comment`) | ❌ No | ✅ Yes |
| **Used In** | Config files, DevOps, ML | APIs, Web apps | Documents, Legacy systems |

---

## 🔹 Where is YAML Used? (Real-World Examples)

### 1️⃣ **Machine Learning Pipelines (MLflow, Hydra)**
```yaml
experiment:
  name: "Image Classification"
  parameters:
    learning_rate: 0.01
    batch_size: 64
```

### 2️⃣ **Kubernetes (Defining Deployments & Services)**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: app-container
      image: my-app:latest
```

### 3️⃣ **GitHub Actions (CI/CD Pipelines)**
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: echo "Deploying Model..."
```

---

## 🔹 YAML in Python
Python has libraries like `PyYAML` to parse and generate YAML files.

### ✅ Install PyYAML
```bash
pip install pyyaml
```

### ✅ Read a YAML File in Python
```python
import yaml

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)
    print(data)
```

### ✅ Write a YAML File in Python
```python
import yaml

data = {
    "name": "John Doe",
    "age": 30,
    "is_student": False,
    "fruits": ["Apple", "Banana", "Orange"]
}

with open("output.yaml", "w") as file:
    yaml.dump(data, file)
```

---

## 🔹 Best Practices for YAML
✔️ **Use Consistent Indentation** – Always use spaces, never tabs  
✔️ **Avoid Deep Nesting** – Too many levels can make YAML hard to read  
✔️ **Use Comments Sparingly** – Comments help but shouldn’t clutter the file  
✔️ **Validate YAML Files** – Use linters like [`yamllint`](https://www.yamllint.com/)  

---

