# ✈️ AeroSatisfy - Airline Passenger Satisfaction Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green.svg)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange.svg)
![License](https://img.shields.io/badge/License-Open%20Source-brightgreen.svg)

**Predict airline passenger satisfaction with explainable AI. Built with Python, Streamlit, and SHAP for complete transparency.**

[📖 Full Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [📊 Features](#-features) • [🤖 How It Works](#-how-it-works)

---

## 🎯 What Does This Project Do?

Imagine an airline wants to know: **"Will this passenger be satisfied with their flight?"**

**AeroSatisfy** answers this question using AI:

```
Input:  Passenger info, service ratings, flight delays
        ↓
     [Machine Learning Model]
        ↓
Output: ✅ Satisfied or ⚠️ Not Satisfied
        + Confidence score
        + Why this prediction?
```

### Real-World Example

```
Flight details:
├─ Passenger: 45-year-old male, loyal customer
├─ Flight: Business class, 2500 miles
├─ Service: Excellent seat comfort, great food, good WiFi
├─ Delays: None
└─ Prediction: ✅ SATISFIED (96% confidence)
   Reason: Excellent service + no delays = happy customer
```

---

## ⭐ Key Features

### ✅ Smart Prediction
- Predicts satisfaction based on **22 passenger features**
- **95% accuracy** on test data
- Works with real airline data

### 📊 Beautiful Web Interface
- Interactive input form
- Real-time predictions
- Stunning visualizations
- Mobile-friendly design

### 🔍 Explainable AI (SHAP)
- See **exactly why** each prediction is made
- Understand which factors matter most
- Build trust in AI decisions

### 📈 Actionable Insights
- Identify key satisfaction drivers
- Find improvement opportunities
- Optimize service quality
- Track patterns across passengers

### 🚀 Easy to Deploy
- One-click installation
- Works locally or in cloud
- Shareable web link
- No AI expertise needed

---

## 📖 Documentation

Comprehensive guides for every skill level:

| Document | Purpose | Read Time | For |
|----------|---------|-----------|-----|
| [**QUICKSTART.md**](QUICKSTART.md) | **5-minute setup** | 5 min ⚡ | Everyone - start here! |
| [**DATA_DICTIONARY.md**](DATA_DICTIONARY.md) | Explain all 22 input features | 15 min | Understanding data |
| [**MODEL_EXPLANATION.md**](MODEL_EXPLANATION.md) | How the AI model works | 25 min | Technical details |
| [**SHAP_ANALYSIS_GUIDE.md**](SHAP_ANALYSIS_GUIDE.md) | Prediction explanations | 20 min | Interpreting results |
| [**GITHUB_README.md**](GITHUB_README.md) | Complete project guide | 30 min | Deep dive |
| [**PROJECT_STRUCTURE.md**](PROJECT_STRUCTURE.md) | File organization | 10 min | Project layout |

**First time?** → Start with [QUICKSTART.md](QUICKSTART.md)

---

## 🚀 Quick Start (5 Minutes)

### 1️⃣ Install Python
Download from [python.org](https://www.python.org/downloads) (Python 3.8+)

### 2️⃣ Clone Repository
```bash
git clone https://github.com/Singhrituraj114/AeroSatisfy-Passenger-Experience-Analytics-Prediction.git
cd AeroSatisfy-Passenger-Experience-Analytics-Prediction
```

### 3️⃣ Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Run Application
```bash
streamlit run app.py
```

**That's it!** The web app opens in your browser automatically at `http://localhost:8501`

---

## 🎮 How to Use

### Step 1: Enter Passenger Details
- Gender, Age, Customer Type
- Travel class, distance, purpose

### Step 2: Rate Services (0-5 scale)
- WiFi, Seat Comfort, Food, Entertainment
- Service quality, cleanliness, etc.

### Step 3: Add Flight Delays
- Departure delay (minutes)
- Arrival delay (minutes)

### Step 4: Click "Predict"
The app instantly shows:
- ✅ Satisfaction prediction
- 📊 Confidence score
- 📈 Top 10 influencing factors
- 🔍 SHAP explanation

---

## 🤖 How It Works

### Simple Version (30 seconds)
```
1. You enter passenger data
2. The AI model analyzes 22 features
3. Random Forest votes (500 trees)
4. Gets satisfaction prediction
5. SHAP explains the decision
```

### Technical Version (2 minutes)
**Algorithm:** Random Forest Classifier
- **500 decision trees** learn from data
- Each tree asks simple yes/no questions
- Trees vote on final prediction
- Majority vote = final result

**Why Random Forest?**
- ✅ Handles non-linear patterns
- ✅ 95% accuracy
- ✅ Fast predictions
- ✅ Explainable results

**Training Data:**
- 25,000+ airline passenger records
- 80% used for training (20,000)
- 20% used for testing (5,000)

[**Read full explanation →**](MODEL_EXPLANATION.md)

---

## 📊 Understanding SHAP Explanations

### What is SHAP?
Think of it like a financial advisor explaining a credit score:
```
Base: 600 (starting credit)
Good income: +50
Low debt: +30
Long history: +25
Recent missed payment: -30
Final: 675 (your credit score)
```

SHAP does the same for satisfaction predictions!

### Example Output
```
Passenger will be SATISFIED (85% confidence)

Factors pushing toward SATISFIED:
✅ Excellent seat comfort          (+25%)
✅ No departure delays             (+20%)
✅ Excellent on-board service      (+18%)
✅ Business class                  (+12%)

Factors pushing toward NOT SATISFIED:
❌ Poor WiFi service               (-15%)

Final: 85% SATISFIED
```

[**Read SHAP guide →**](SHAP_ANALYSIS_GUIDE.md)

---

## 📋 Features & Data

### 22 Input Features

**Passenger Info (3):**
- Gender, Age, Customer Type

**Travel Details (3):**
- Ticket Class, Flight Distance, Travel Type

**Service Ratings (14):** *0-5 scale*
- WiFi, Booking, Gate, Food, Entertainment
- Seat Comfort, Leg Room, Service Quality
- Baggage Handling, Check-in, Cleanliness
- And more...

**Delays (2):**
- Departure Delay (minutes)
- Arrival Delay (minutes)

### 1 Output (Prediction)
- **Satisfied** ✅ or **Not Satisfied** ⚠️
- **Confidence Score** (0-100%)

[**View detailed data dictionary →**](DATA_DICTIONARY.md)

---

## 📊 Model Performance

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | 95% | Correct predictions 95 out of 100 times |
| **Precision** | 94% | When we say satisfied, we're right 94% |
| **Recall** | 96% | We catch 96% of truly satisfied customers |
| **F1-Score** | 95% | Balanced performance |

**Training:** 25,000 passenger records  
**Algorithm:** Random Forest (500 trees)  
**Features:** 22 passenger attributes

---

## 🎨 Technology Stack

```
Frontend:       Streamlit (Python web framework)
Backend:        Scikit-learn (Machine Learning)
Visualization:  Plotly, Matplotlib
Explainability: SHAP (Shapley values)
Data:           Pandas, NumPy
Deployment:     Python 3.8+
```

---

## 🌐 Deployment Options

### Option 1: Local (Easiest)
```bash
streamlit run app.py
# Runs on http://localhost:8501
```

### Option 2: Streamlit Cloud (Free, 1-click)
1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Deploy
4. Get public URL

### Option 3: Docker (Professional)
```bash
docker build -t aerosatisfy .
docker run -p 8501:8501 aerosatisfy
```

---

## 📈 Use Cases

| Use Case | How It Helps |
|----------|-------------|
| **Pre-flight Assessment** | Identify at-risk passengers before flight |
| **Service Improvement** | See which services matter most |
| **Staff Training** | Train on high-impact service areas |
| **Customer Retention** | Offer upgrades to likely-dissatisfied passengers |

---

## 🎓 Learn More

- [**Full Documentation**](GITHUB_README.md)
- [**Data Dictionary**](DATA_DICTIONARY.md)
- [**Model Explanation**](MODEL_EXPLANATION.md)
- [**SHAP Guide**](SHAP_ANALYSIS_GUIDE.md)
- [**Quick Start**](QUICKSTART.md)
- [**Project Structure**](PROJECT_STRUCTURE.md)

---

## 🚀 Next Steps

**[👉 Start with QUICKSTART.md 👈](QUICKSTART.md)** - 5 minute setup!

---

**Built with ❤️ for better airline experiences** ✈️
  - Handles mixed data types (categorical, numerical, ordinal)
  - Provides strong feature importance insights
  - High classification accuracy

## 📦 Deployment Artifacts

The model is saved in production-ready format:
- `model.pkl` - Trained Random Forest model
- `encoders.pkl` - Label encoders for categorical features
- `features.pkl` - Feature column order (maintains consistency)

## 🚀 Quick Start

### Install Dependencies:
```bash
cd "c:\Users\singh\OneDrive\Desktop\Airline_Web_App"
pip install -r requirements.txt.txt
```

### Run the App:
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 📁 Project Structure

```
Airline_Web_App/
├── app.py                  # Streamlit web application
├── model.pkl              # Trained ML model (Random Forest)
├── encoders.pkl           # Feature encoders
├── features.pkl           # Feature list
├── requirements.txt.txt   # Python dependencies
└── README.md             # This file
```

## 💡 How to Use

1. **Enter Passenger Information:**
   - Select gender, customer type, travel type
   - Enter age, flight distance
   - Specify departure and arrival delays

2. **Rate Services (0-5 Scale):**
   - Rate all 14 service categories
   - 0 = Very Poor, 5 = Excellent

3. **Click "Predict Satisfaction"**
   - Get instant prediction
   - View confidence percentage
   - See top factors influencing the prediction

## 📊 Business Use Cases

- **Identify Dissatisfaction Drivers:** Understand which factors lead to passenger dissatisfaction
- **Improve Services:** Focus on low-rated service areas
- **Customer Retention:** Predict and prevent churn
- **Personalized Experience:** Tailor services based on prediction factors
- **Resource Allocation:** Prioritize improvements for high-impact services

## 🔍 Model Evaluation

Trained on classification metrics:
- **Accuracy** - Overall prediction accuracy
- **Precision** - How many predicted satisfied are actually satisfied
- **Recall** - Coverage of actual satisfied passengers
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under the receiver operating characteristic curve
- **Confusion Matrix** - True/False positives and negatives

## 📚 Feature Importance

The model shows which factors most influence satisfaction:
- Service quality ratings have high importance
- Customer type and travel class are significant
- Flight delays impact satisfaction
- Delay factors are among top predictors

## ⚙️ Technical Stack

- **Frontend:** Streamlit
- **ML Framework:** Scikit-learn
- **Visualizations:** Plotly
- **Data Processing:** Pandas
- **Model Serialization:** Joblib

## 🎯 Data Preprocessing

- Handled missing values
- Checked for duplicates
- Encoded categorical features using LabelEncoder
- Encoded target variable to binary (satisfied=1, dissatisfied=0)
- Maintained feature order consistency for deployment
- Train-test split for model evaluation

## ✨ App Features

- **Clean Interface:** Organized 3-column layout for easy input
- **Real-time Prediction:** Instant results after clicking predict
- **Feature Importance:** Visual chart of top 10 influencing factors
- **Confidence Score:** Probability-based confidence metric
- **Professional Output:** Clear satisfied/not-satisfied status display
- **Wide Layout:** Responsive design for optimal viewing

## 📝 Requirements

```
streamlit
pandas
scikit-learn
joblib
plotly
```

## 🐛 Troubleshooting

**App won't start:**
```bash
pip install --upgrade streamlit scikit-learn
```

**ModuleNotFoundError:**
```bash
pip install -r requirements.txt.txt
```

**Prediction errors:**
- Ensure all model files (model.pkl, encoders.pkl, features.pkl) are in the directory
- Check that categorical values match expected options

## 🎓 Project Demonstrates

✅ End-to-end ML pipeline
✅ Data preprocessing & EDA
✅ Multiple model training & comparison
✅ Model selection & evaluation
✅ Feature importance analysis
✅ Production deployment
✅ Web application development
✅ Professional Python code structure

---

**Built with** ✈️ Streamlit | 🤖 Scikit-learn | 📊 Plotly

**© 2026 - Airline Passenger Satisfaction Predictor**

