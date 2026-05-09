# 🚀 Quick Start Guide - Getting Started in 5 Minutes

## Prerequisites
You need:
- **Python 3.8 or higher** (check by running `python --version`)
- **pip** (Python package manager - comes with Python)
- **Git** (optional, for cloning the repository)
- **Internet connection** (to download packages)

---

## Installation Steps

### Step 1: Clone or Download the Repository

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/Singhrituraj114/AeroSatisfy-Passenger-Experience-Analytics-Prediction.git
cd AeroSatisfy-Passenger-Experience-Analytics-Prediction
```

**Option B: Direct Download**
1. Click the green "Code" button on GitHub
2. Click "Download ZIP"
3. Extract the ZIP file
4. Open Terminal/Command Prompt and navigate to the folder:
   ```bash
   cd path/to/extracted/folder
   ```

---

### Step 2: Create a Virtual Environment

**Why?** Virtual environments keep your project's packages isolated from your system.

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**You'll know it works when you see `(venv)` at the start of your terminal line.**

---

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

This installs all 8 required packages:
- `streamlit` - Web app framework
- `pandas` - Data processing
- `scikit-learn` - Machine learning
- `joblib` - Model loading
- `matplotlib` - Charts
- `plotly` - Interactive visualizations
- `shap` - Model explanations
- `numpy` - Numerical computing

**Expected time:** 2-5 minutes

---

### Step 4: Run the Application

```bash
streamlit run app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**Your browser will automatically open.** If not, manually go to `http://localhost:8501`

---

## 🎉 You're Done!

The web app is now running. You should see:
- 🎨 Beautiful header with gradient colors
- 📝 Input form for passenger details
- 🎯 Prediction button
- 📊 Results with visualizations and SHAP analysis

---

## Common Issues & Solutions

### Issue 1: "Python is not recognized"
**Solution:**
- Python might not be installed
- Download from [python.org](https://www.python.org/downloads)
- Make sure to check "Add Python to PATH" during installation

### Issue 2: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:**
- You forgot to activate the virtual environment
- Run: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
- Then run: `pip install -r requirements.txt`

### Issue 3: "No module named 'model', 'encoders', or 'features'"
**Solution:**
- The .pkl files might be missing
- Check that `model.pkl`, `encoders.pkl`, and `features.pkl` are in the same folder as `app.py`
- If missing, you need to train the model (see "Training the Model" section below)

### Issue 4: "Address already in use"
**Solution:**
- Another process is using port 8501
- Kill the existing process or use a different port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

### Issue 5: The app opens but shows blank/error page
**Solution:**
- Wait 10-15 seconds for all components to load
- Check terminal for error messages
- Try refreshing the browser (F5)
- Clear browser cache

---

## How to Use the Web App (Step-by-Step)

### 1. **Fill in Passenger Information**
```
Gender:         Select Male or Female
Age:            Enter a number (18-80)
Customer Type:  Select Loyal or Disloyal
```

### 2. **Enter Travel Details**
```
Ticket Class:       Select Economy / Eco Plus / Business
Flight Distance:    Enter distance in miles (100-5000)
Travel Type:        Select Business or Personal
```

### 3. **Rate 14 Services (0-5 Scale)**
Move sliders for each service:
- 0 = Not Applicable
- 1 = Poor
- 2 = Fair
- 3 = Good
- 4 = Very Good
- 5 = Excellent

**Services to rate:**
1. Inflight WiFi
2. Departure/Arrival Time
3. Online Booking
4. Gate Location
5. Food and Drink
6. Online Boarding
7. Seat Comfort
8. Entertainment
9. On-board Service
10. Leg Room
11. Baggage Handling
12. Check-in
13. Inflight Service
14. Cleanliness

### 4. **Enter Delays**
```
Departure Delay:    Minutes delayed (0-120+)
Arrival Delay:      Minutes delayed (0-120+)
```

### 5. **Get Prediction**
Click the big button: **🚀 Predict Satisfaction**

### 6. **View Results**
The app displays:
- ✅ **Prediction** - Satisfied or Not Satisfied
- 📊 **Confidence** - How sure (0-100%)
- 📈 **Top 10 Factors** - Most important features
- 🔍 **SHAP Analysis** - Why this prediction
- 📋 **Probabilities** - Exact percentages

---

## Understanding Results

### Example Results:

```
🎯 PREDICTION RESULT

✅ Predicted Class: SATISFIED
📊 Confidence: 87%

Explanation:
This passenger is 87% likely to be SATISFIED based on:
- Excellent seat comfort (+25%)
- No departure delay (+15%)
- Excellent on-board service (+18%)
- Good food and drink (+12%)
- One negative: Fair WiFi (-8%)

Top Influencing Factors:
1. Seat Comfort        ⭐⭐⭐⭐⭐
2. On-board Service    ⭐⭐⭐⭐
3. Departure Delay     ⭐⭐⭐⭐
4. Food Quality        ⭐⭐⭐
...
```

### What the Percentages Mean:
- **90%+ = Highly Confident** - Reliable prediction
- **75-90% = Confident** - Good prediction
- **60-75% = Moderate Confidence** - Reasonably reliable
- **50-60% = Low Confidence** - Borderline, could go either way

---

## 📊 Making Predictions: Complete Examples

### Example 1: First-Time Economy Passenger

**Inputs:**
```
Gender: Female
Age: 28
Customer Type: Disloyal (first time flying this airline)
Class: Economy
Distance: 500 miles (short flight)
Travel Type: Personal

Services: All rated 3/5 (Good - average experience)

Delays: 
  Departure: 20 minutes
  Arrival: 15 minutes
```

**Expected Result:**
```
⚠️ NOT SATISFIED (62% confidence)
Reason: First-time flyer, short flight, small delays add up, 
        average services don't impress new customers
```

---

### Example 2: Loyal Business Traveler

**Inputs:**
```
Gender: Male
Age: 52
Customer Type: Loyal (frequent flyer)
Class: Business
Distance: 2500 miles (long flight)
Travel Type: Business

Services:
  Seat Comfort: 5/5
  WiFi: 5/5
  Food: 4/5
  Service: 5/5
  Entertainment: 4/5
  (others: 4-5)

Delays: 0 minutes
```

**Expected Result:**
```
✅ SATISFIED (96% confidence)
Reason: Loyal customer, premium service, excellent ratings,
        no delays = very satisfied
```

---

### Example 3: Troubled Flight

**Inputs:**
```
Gender: Female
Age: 35
Customer Type: Loyal
Class: Economy Plus
Distance: 1800 miles
Travel Type: Business

Services:
  Seat Comfort: 2/5 (cramped)
  WiFi: 1/5 (didn't work)
  Food: 2/5 (poor quality)
  Service: 3/5 (okay)
  Entertainment: 1/5 (limited options)
  (others: 2-3)

Delays: 
  Departure: 45 minutes
  Arrival: 90 minutes
```

**Expected Result:**
```
⚠️ NOT SATISFIED (78% confidence)
Reason: Multiple service issues (WiFi, food, entertainment),
        significant delays, cramped seat on long flight.
        Even loyal customer would be frustrated.
```

---

## 🔧 Training Your Own Model (Advanced)

If you want to retrain the model with new data:

### Requirements:
- A CSV file with passenger data
- All 22 feature columns (see DATA_DICTIONARY.md)
- A "Satisfaction" column (target) with values: "Satisfied" or "Not Satisfied"

### Steps:
1. Prepare your CSV file
2. Create a Python script called `train_model.py`
3. Run: `python train_model.py`
4. New model.pkl will be created
5. Run the app: `streamlit run app.py`

### Sample Training Code:
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('your_data.csv')

# Separate features and target
X = df.drop('Satisfaction', axis=1)
y = df['Satisfaction']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')

print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

---

## 📱 Running on Different Computers

### Windows:
```bash
# Activate
venv\Scripts\activate

# Run
streamlit run app.py
```

### Mac/Linux:
```bash
# Activate
source venv/bin/activate

# Run
streamlit run app.py
```

---

## 🌐 Sharing the App Online

### Option 1: Streamlit Cloud (Free, Easy)
1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy in 2 clicks
5. Get a public URL

### Option 2: Heroku (Free, More Complex)
1. Add Procfile and setup.sh
2. Connect to Heroku
3. Deploy
4. Get a public URL

### Option 3: Docker (Professional)
1. Create Dockerfile
2. Build image
3. Deploy to any cloud (AWS, GCP, Azure)

---

## 🎓 Learning Resources

- **Streamlit Documentation:** https://docs.streamlit.io
- **SHAP Documentation:** https://shap.readthedocs.io
- **Machine Learning Basics:** https://scikit-learn.org
- **Data Science Tutorial:** https://kaggle.com/learn

---

## ✅ Checklist: Am I Ready?

- [ ] Python 3.8+ installed
- [ ] Repository cloned/downloaded
- [ ] Virtual environment created and activated
- [ ] requirements.txt installed
- [ ] model.pkl, encoders.pkl, features.pkl files present
- [ ] app.py file present
- [ ] No error messages in terminal
- [ ] Browser opened to localhost:8501
- [ ] Web app is displaying correctly

**If all checked:** You're ready to make predictions! 🎉

---

## Getting Help

### If Something Breaks:

1. **Read error message carefully** - It usually tells you the problem
2. **Check the terminal** - Scroll up to see the full error
3. **Search GitHub Issues** - Someone might have had the same problem
4. **Restart from scratch:**
   ```bash
   # Kill the app (Ctrl+C)
   # Deactivate virtual env
   deactivate
   # Delete venv folder
   # Start over from Step 2
   ```

### Common Commands:

```bash
# Activate virtual environment
source venv/bin/activate              # Mac/Linux
venv\Scripts\activate                 # Windows

# Install packages
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Deactivate virtual environment
deactivate

# Check Python version
python --version

# List installed packages
pip list

# Upgrade a package
pip install --upgrade streamlit
```

---

## Next Steps

1. ✅ Install and run the app
2. 📖 Read DATA_DICTIONARY.md to understand features
3. 🔍 Read SHAP_ANALYSIS_GUIDE.md to understand predictions
4. 📊 Make predictions and analyze results
5. 🔧 Customize or extend the project

---

## 🎉 Congratulations!

You now have a fully functional AI-powered airline satisfaction prediction system.

**Happy predicting!** ✈️

---

**Questions?** Check the main README.md or GitHub Issues
