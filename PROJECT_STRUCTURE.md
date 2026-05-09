# 📁 Project Structure & File Guide

## Directory Overview

```
AeroSatisfy-Passenger-Experience-Analytics-Prediction/
│
├── 📄 app.py                    # Main web application (Streamlit)
├── 🤖 model.pkl                 # Trained Random Forest model
├── 🔑 encoders.pkl              # Data encoders for categorical features
├── 📊 features.pkl              # List of expected features
│
├── 📚 Documentation Files:
│   ├── README.md                # ← START HERE (main overview)
│   ├── GITHUB_README.md         # Detailed GitHub version
│   ├── QUICKSTART.md            # Installation & getting started (5 min)
│   ├── DATA_DICTIONARY.md       # Explain all 22 features
│   ├── MODEL_EXPLANATION.md     # How the AI model works
│   ├── SHAP_ANALYSIS_GUIDE.md   # Understanding predictions
│   └── PROJECT_STRUCTURE.md     # This file
│
├── 📦 requirements.txt          # Python dependencies
├── 🐍 runtime.txt               # Python version
│
└── 🎨 Other files (optional):
    ├── .gitignore              # Git ignore rules
    ├── train_model.py          # Script to retrain model (if available)
    └── sample_data.csv         # Sample data for testing (if available)
```

---

## File Descriptions

### 🎯 Core Application Files

#### **app.py** (Main Application)
- **What it does:** Runs the entire web application
- **Built with:** Streamlit framework
- **Size:** ~1000-3000 lines of code
- **Key functions:**
  - Creates user interface
  - Loads model and data encoders
  - Processes user input
  - Makes predictions
  - Generates visualizations
  - Calculates SHAP explanations
- **You should:** Run this with `streamlit run app.py`
- **Don't edit unless:** You want to modify the interface or add features

#### **model.pkl** (Trained Machine Learning Model)
- **What it is:** Binary file containing trained Random Forest Classifier
- **Size:** ~50-200 MB (depends on model complexity)
- **Contains:** 500 decision trees learned from 25,000+ passenger records
- **How to use:** Automatically loaded by app.py
- **Don't edit directly:** It's a binary file - edit would corrupt it
- **To update:** Retrain using your own data (see training script)

#### **encoders.pkl** (Data Preprocessing Rules)
- **What it contains:** Rules to convert text to numbers
- **Examples:**
  - Male → 1, Female → 0
  - Business → 2, Economy → 0, Eco Plus → 1
  - Loyal → 1, Disloyal → 0
- **Used by:** app.py when processing user input
- **Size:** ~1-5 KB (very small)
- **Why needed:** Model was trained on numbers; encoders convert text inputs

#### **features.pkl** (Feature List)
- **What it contains:** Ordered list of all 22 expected features
- **Example content:** `['Age', 'Gender', 'Customer_Type', 'Class', ...]`
- **Purpose:** Ensures model receives features in correct order
- **Size:** ~1 KB
- **Used by:** app.py for validation and preprocessing

---

### 📚 Documentation Files (Read These!)

#### **README.md** (Main Overview)
- **Purpose:** First file people see on GitHub
- **Contains:**
  - Project summary
  - Key features
  - How to install
  - Quick example
- **Read time:** 5-10 minutes
- **Start here!** ✅

#### **GITHUB_README.md** (Detailed Version)
- **Purpose:** Comprehensive project documentation
- **Contains:**
  - Real-world use cases
  - Complete feature explanations
  - Data flow diagrams
  - Example predictions
  - Technical details
- **Read time:** 20-30 minutes
- **For:** Understanding every aspect of the project

#### **QUICKSTART.md** (Getting Started Guide)
- **Purpose:** Step-by-step installation
- **Contains:**
  - System requirements
  - 5-minute installation steps
  - How to use the web app
  - Troubleshooting
  - Example predictions
- **Read time:** 5-15 minutes
- **For:** Developers want to get running immediately

#### **DATA_DICTIONARY.md** (Feature Explanations)
- **Purpose:** Explain every single input field
- **Contains:**
  - All 22 features explained
  - Ranges and valid values
  - Why each feature matters
  - Example data point
- **Read time:** 15-20 minutes
- **For:** Understanding data structure

#### **MODEL_EXPLANATION.md** (Technical Deep Dive)
- **Purpose:** Explain the machine learning model
- **Contains:**
  - What is Random Forest
  - Training process
  - How predictions work
  - Feature importance
  - Performance metrics
  - Limitations
  - How to improve
- **Read time:** 25-35 minutes
- **For:** Data scientists, developers, curious minds

#### **SHAP_ANALYSIS_GUIDE.md** (Prediction Explanation)
- **Purpose:** Understand why model makes predictions
- **Contains:**
  - SHAP concept explanation
  - How to read SHAP plots
  - Real-world examples
  - Common patterns
  - FAQ
- **Read time:** 20-25 minutes
- **For:** Understanding individual predictions

#### **PROJECT_STRUCTURE.md** (This File)
- **Purpose:** Guide to all project files
- **Contains:** What each file does, how to use it

---

### 📦 Configuration Files

#### **requirements.txt**
- **Purpose:** List of Python packages needed
- **Format:** One package per line with version
- **Contents:**
  ```
  streamlit
  pandas
  scikit-learn
  joblib
  matplotlib
  plotly
  shap
  numpy
  ```
- **Use:** `pip install -r requirements.txt`

#### **runtime.txt**
- **Purpose:** Specify Python version
- **Contents:** e.g., `python-3.9.13`
- **Used by:** Deployment platforms (Heroku, Streamlit Cloud)

---

## How Files Work Together

### The Data Flow

```
User Input (Web Form)
        ↓
    app.py
        ↓
encoders.pkl (Convert text to numbers)
        ↓
features.pkl (Arrange in correct order)
        ↓
model.pkl (Make prediction)
        ↓
SHAP calculations
        ↓
Visualizations & Results
        ↓
Display in Web Browser
```

### File Dependencies

```
app.py depends on:
├─ model.pkl (the trained model)
├─ encoders.pkl (to encode inputs)
├─ features.pkl (to validate features)
└─ requirements.txt (to know what to install)

Documentation files are independent:
├─ README.md
├─ QUICKSTART.md
├─ DATA_DICTIONARY.md
├─ MODEL_EXPLANATION.md
├─ SHAP_ANALYSIS_GUIDE.md
└─ PROJECT_STRUCTURE.md (you are here)
```

---

## Reading Guide (What to Read First?)

### For Quick Start (20 minutes)
1. ✅ **README.md** - Understand what this is
2. ✅ **QUICKSTART.md** - Install and run
3. Make a few predictions
4. Done!

### For Complete Understanding (2-3 hours)
1. ✅ **README.md** - Overview
2. ✅ **GITHUB_README.md** - Full details
3. ✅ **DATA_DICTIONARY.md** - All features
4. ✅ **MODEL_EXPLANATION.md** - How model works
5. ✅ **SHAP_ANALYSIS_GUIDE.md** - Prediction explanation
6. ✅ **QUICKSTART.md** - Technical setup

### For Data Scientists (Technical)
1. ✅ **MODEL_EXPLANATION.md** - Model architecture
2. ✅ **DATA_DICTIONARY.md** - Feature engineering
3. ✅ **SHAP_ANALYSIS_GUIDE.md** - Model interpretation
4. ✅ Check app.py code for implementation

### For Business Users
1. ✅ **README.md** - What can this do?
2. ✅ **GITHUB_README.md** - Real-world use cases
3. ✅ **SHAP_ANALYSIS_GUIDE.md** - Business insights
4. ✅ **QUICKSTART.md** - How to use

---

## File Modification Guidelines

### Safe to Modify ✅
- **app.py** - Add features, change UI, customize
- **requirements.txt** - Add/remove packages
- Add new documentation files as needed

### Don't Modify ⚠️
- **model.pkl** - Binary file (will corrupt)
- **encoders.pkl** - Binary file (will corrupt)
- **features.pkl** - Binary file (will corrupt)

**To update models:**
Retrain using your own data with training script, not manual editing

---

## File Sizes (Expected)

| File | Size | Notes |
|------|------|-------|
| app.py | 10-50 KB | Python code |
| model.pkl | 50-200 MB | Trained model |
| encoders.pkl | 1-10 KB | Encoding rules |
| features.pkl | 1 KB | Feature list |
| requirements.txt | 0.5 KB | Dependencies |
| Documentation | 500 KB - 1 MB | Markdown files |
| **Total** | **~100-250 MB** | Model is biggest |

---

## Adding New Files

### To Add a New Feature

Create `new_feature.py`:
```python
import streamlit as st

def new_feature():
    st.header("My New Feature")
    # Your code here

if __name__ == "__main__":
    new_feature()
```

Import in app.py:
```python
from new_feature import new_feature
# Call the function at appropriate place
new_feature()
```

### To Add Documentation

Create `TOPIC_NAME.md`:
```markdown
# Topic Title

Explain the topic in simple words...

## Section 1
Details...

## Section 2
More details...
```

---

## Version Control (Git)

### .gitignore Recommendations
```
# Large files
*.pkl
*.joblib
*.pickle

# Virtual environment
venv/
env/
.venv/

# Cache
__pycache__/
*.pyc
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/
```

### Git Workflow
```bash
# Clone
git clone <repo-url>

# Create branch
git checkout -b feature/new-feature

# Make changes
git add app.py
git commit -m "Add new feature"

# Push
git push origin feature/new-feature

# Create Pull Request on GitHub
```

---

## Deployment Files (Optional)

### Streamlit Cloud (streamlit_config.toml)
```toml
[server]
port = 8501
headless = true
```

### Heroku (Procfile)
```
web: streamlit run app.py --server.port=$PORT
```

### Docker (Dockerfile)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

---

## File Naming Conventions

### Python Files
- Use snake_case: `train_model.py`, `data_processor.py`
- Descriptive names: `preprocess.py` (good), `p.py` (bad)

### Markdown Files
- Use UPPERCASE: `README.md`, `DATA_DICTIONARY.md`
- Use underscores: `MODEL_EXPLANATION.md`

### Binary Files
- Use .pkl extension: `model.pkl`, `encoders.pkl`
- Descriptive names: `model_v2.pkl` (for versions)

---

## Troubleshooting File Issues

### Issue: "ModuleNotFoundError"
**Cause:** Missing package in requirements.txt
**Fix:** `pip install -r requirements.txt`

### Issue: "FileNotFoundError: model.pkl"
**Cause:** Model file missing or in wrong directory
**Fix:** Check file exists in same folder as app.py

### Issue: "PermissionError"
**Cause:** File permissions issue
**Fix:**
```bash
# Windows
icacls filename /grant Users:F

# Mac/Linux
chmod 644 filename
```

### Issue: Files too large for GitHub
**Limit:** 100 MB per file
**Solution:** Use Git LFS
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add model.pkl
git commit -m "Add large model file"
```

---

## Backup & Recovery

### Backing Up Important Files
```bash
# Backup model
cp model.pkl model_backup.pkl

# Backup entire project
tar -czf project_backup.tar.gz .
```

### Recovering Files
```bash
# Restore from backup
cp model_backup.pkl model.pkl

# Restore from git history
git checkout HEAD -- model.pkl
```

---

## Performance Optimization

### If app is slow:

1. **Check app.py:**
   ```python
   # Use caching
   @st.cache_resource
   def load_model():
       # Load once, reuse many times
   ```

2. **Check model.pkl:**
   - Too large → Use quantization/compression
   - Complex → Use faster model

3. **Check SHAP calculation:**
   - Very slow → Cache SHAP values
   - Use sampling instead of full calculation

---

## File Size Reduction

### Compress Model
```python
# Before: 150 MB
# After: 50 MB

from sklearn.tree import DecisionTreeClassifier
# Use shallow trees instead of deep
model = RandomForestClassifier(max_depth=10)
```

### Remove Unnecessary Files
- Delete old_model_v1.pkl
- Delete test_data.csv if large
- Compress images/visualizations

---

## Documentation Best Practices

### Writing New Docs
1. **Title:** Clear, concise (H1 #)
2. **Overview:** 2-3 sentence summary
3. **Table of Contents:** For long docs
4. **Examples:** Code snippets, outputs
5. **Key Takeaways:** Summary box at end

### Code Comments
```python
# Good comment
new_value = old_value * 2  # Double the age

# Bad comment
x = y * 2  # This multiplies
```

---

## File Organization Tips

### Recommended Structure (Advanced)
```
project/
├── src/
│   ├── app.py
│   ├── preprocessing.py
│   └── models.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── models/
│   ├── model.pkl
│   ├── encoders.pkl
│   └── features.pkl
├── docs/
│   ├── README.md
│   ├── QUICKSTART.md
│   └── ...
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
└── requirements.txt
```

---

## Summary

### Key Files:
- **app.py** - The application
- **model.pkl** - The AI brain
- **Documentation** - How to use everything

### Before You Start:
- ✅ Read README.md
- ✅ Read QUICKSTART.md
- ✅ Run `pip install -r requirements.txt`
- ✅ Run `streamlit run app.py`

### For Questions:
- 📖 Check DATA_DICTIONARY.md
- 🤖 Check MODEL_EXPLANATION.md
- 🔍 Check SHAP_ANALYSIS_GUIDE.md

---

**Now you understand the project structure! 🎉**
