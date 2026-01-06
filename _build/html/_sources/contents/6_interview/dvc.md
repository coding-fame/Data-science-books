# DVC (Data Version Control)

## What is DVC?
**Data Version Control (DVC)** is an open-source tool that:
- Manages large datasets/models alongside code
- Creates reproducible ML pipelines
- Tracks experiments and metrics
- Integrates with Git for version control

### **1. Install DVC**
```bash
pip install dvc
```

---

### Core Benefits
| Feature | Description |
|---------|-------------|
| **Data Versioning** | Track dataset/model versions without Git LFS |
| **Pipeline Automation** | Define and reproduce complex workflows |
| **Experiment Tracking** | Compare hyperparameters and metrics |
| **Cloud Storage** | Works with S3, GCP, Azure, and local storage |

---


## 2. Pipeline Setup Guide

### Step 1: Repository Initialization
```bash
git init
dvc init
git commit -m "Initialize DVC"
```

### Step 2: Project Structure
```
├── .gitignore
├── params.yaml
├── dvc.yaml
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── data/          # Added to .gitignore
├── models/        # Added to .gitignore
└── reports/       # Added to .gitignore
```

### Step 3: Pipeline Configuration (`dvc.yaml`)
```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw
      - src/preprocess.py
    outs:
      - data/processed

  train:
    cmd: python src/train.py
    deps:
      - data/processed
      - src/train.py
    outs:
      - models/model.pkl
```

### Step 4: Parameter Management (`params.yaml`)
```yaml
data_ingestion:
  test_size: 0.2
  
feature_engineering:
  max_features: 1000

model_building:
  learning_rate: 0.01
  epochs: 100
```

### Step 5: Execute Pipeline
```bash
dvc repro  # Builds pipeline based on dvc.yaml
dvc dag    # Visualize pipeline structure
```

---

## 3. Experiment Management

### Running Experiments
```bash
dvc exp run --set-param model_building.learning_rate=0.001
```

### Tracking Metrics with DVCLive
```python
from dvclive import Live

with Live(save_dvc_exp=True) as live:
    live.log_metric("accuracy", 0.92)
    live.log_params(params)  # From params.yaml
```

### Experiment Operations
```bash
dvc exp show    # List all experiments
dvc exp diff    # Compare experiment metrics
dvc exp apply   # Restore specific experiment
```

---


---

## 4. Key Concepts & Components

### DVC Files
| File | Purpose |
|------|---------|
| `.dvc` | Pointer files for large data/model versioning |
| `dvc.yaml` | Pipeline definition file |
| `params.yaml` | Central configuration for parameters |

### Storage Strategy
```
Local Repository
├── .git/        # Code versioning
└── .dvc/
    ├── cache/   # Data/model versioning
    └── config   # Remote storage config

Remote Storage ←→ DVC Cache (Synchronized via dvc push/pull)
```

---

## 5. Command Reference

### Basic Workflow
```bash
dvc add data/raw          # Track dataset
dvc push                  # Sync with remote
dvc pull                  # Retrieve updates
dvc repro                 # Rebuild pipeline
```

### Experiment Control
```bash
dvc exp run               # Start new experiment
dvc exp show --only-changed  # Display metric changes
dvc exp branch            # Create Git branch from experiment
```

### Data Management
```bash
dvc gc                    # Cleanup unused data
dvc get https://github.com/...  # Import datasets
dvc list                  # View remote storage
```

---

## 6. Comparison with Alternatives

| Feature                | DVC       | Git-LFS   | MLflow    |
|------------------------|-----------|-----------|-----------|
| Data Versioning        | ✅        | ✅        | ❌        |
| Pipeline Automation    | ✅        | ❌        | Limited   |
| Metric Tracking        | ✅        | ❌        | ✅        |
| Cloud Storage Support  | ✅        | ✅        | ✅        |
| CI/CD Integration      | ✅        | ❌        | ✅        |

---

## **DVC Workflow Summary**

| Step | Command |
|------|---------|
| Initialize DVC | `dvc init` |
| Track data/models | `dvc add <file>` |
| Set up remote storage | `dvc remote add -d <remote-name> <storage-path>` |
| Push data/models | `dvc push` |
| Define pipeline | `dvc.yaml` |
| Run pipeline | `dvc repro` |
| Track metrics | `dvc metrics show` |
| Run experiments | `dvc exp run --set-param <param>=<value>` |
| Compare experiments | `dvc exp diff` |
| Collaborate | `git push`, `dvc push` |

---

## **Best Practices for Using DVC**

### **1. Keep Large Files Out of Git**
- Use `.gitignore` to exclude large datasets and models.
- Track them with DVC instead.

### **2. Regularly Push Changes to Remote Storage**
- Avoid losing data by syncing changes frequently.
- Use `dvc push` after committing to Git.

### **3. Automate Pipelines**
- Use `dvc repro` to keep dependencies up-to-date.
- Integrate with CI/CD workflows for automation.

### **4. Track Experiments Systematically**
- Use `dvc exp` commands to log and compare runs.
- Maintain a `params.yaml` file for easy hyperparameter tuning.

### **5. Collaborate Efficiently**
- Use `dvc pull` to fetch the latest datasets/models.
- Share `.dvc` files in Git while keeping large files in remote storage.

---

## **Conclusion**
DVC simplifies machine learning workflows by integrating **data versioning, experiment tracking, and pipeline automation** into Git-based projects. It ensures reproducibility, facilitates collaboration, and scales efficiently from development to production.

---
