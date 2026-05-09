# ✈️ AeroSatisfy - Airline Passenger Experience Analytics & Prediction

**Predict airline passenger satisfaction using AI-powered machine learning**

A powerful web application that uses advanced machine learning to predict whether an airline passenger will be satisfied or dissatisfied with their flight experience. Built with Python, Streamlit, and SHAP for full model interpretability.

---

## 📌 What is This Project?

This project solves a real-world problem: **How can airlines predict if a passenger will be satisfied?**

Airlines want to identify unhappy customers early so they can improve their service. This machine learning model does exactly that by:
- ✅ Analyzing passenger demographics and flight details
- ✅ Evaluating service ratings across 14 categories
- ✅ Checking flight delays
- ✅ Predicting satisfaction with **confidence scores**
- ✅ Explaining **WHY** each prediction was made (SHAP analysis)

---

## 🎯 Real-World Use Cases

| Scenario | How It Helps |
|----------|-------------|
| **Pre-flight Prediction** | Identify likely dissatisfied customers before they fly to provide better service |
| **Service Improvement** | See which factors matter most for satisfaction |
| **Customer Retention** | Offer upgrades or compensation to at-risk passengers |
| **Resource Allocation** | Prioritize service improvements where they matter most |
| **Training Focus** | Train staff on high-impact service areas |

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────┐
│   User Input Interface (Web)    │  ← Web form for passenger data
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│   Data Processing & Encoding    │  ← Convert inputs to machine format
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│   Machine Learning Model        │  ← Random Forest Classifier
│   (Trained on 25,000+ records)  │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│   Prediction & Confidence       │  ← Satisfied or Not Satisfied
│   + SHAP Explanations           │  ← Why did the model decide this?
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│   Beautiful Visualizations      │  ← Charts, gauges, explanations
└─────────────────────────────────┘
```

---

## 📊 Dataset Overview

**25,000+ airline passenger records** with the following information:

### Passenger Information (3 features)
- **Gender** - Male or Female
- **Age** - Passenger age (numeric)
- **Customer Type** - Loyal Customer or Disloyal Customer

### Travel Information (3 features)
- **Ticket Class** - Economy, Economy Plus, or Business
- **Flight Distance** - Distance traveled in miles
- **Travel Type** - Business or Personal

### Service Quality Ratings (14 features)
Each rated on a scale of **0-5** where:
- **0** = Not Applicable
- **1** = Poor
- **2** = Fair
- **3** = Good
- **4** = Very Good
- **5** = Excellent

**Service Categories:**
1. Inflight WiFi Service
2. Departure/Arrival Time Convenience
3. Ease of Online Booking
4. Gate Location
5. Food and Drink
6. Online Boarding
7. Seat Comfort
8. Inflight Entertainment
9. On-board Service
10. Leg Room Service
11. Baggage Handling
12. Check-in Service
13. Inflight Service
14. Cleanliness

### Flight Delays (2 features)
- **Departure Delay (minutes)** - Minutes delayed at departure
- **Arrival Delay (minutes)** - Minutes delayed at arrival

### Target Variable (What we predict)
- **Satisfaction Status** - Either "Satisfied" ✅ or "Not Satisfied" ⚠️

---

## 🤖 The Machine Learning Model

### What Model Is Used?
**Random Forest Classifier** - Think of it as a collection of decision trees voting on the prediction.

### How Does It Work? (Simple Explanation)

Imagine you're a flight supervisor with 25,000 past passenger records. You want to predict if a new passenger will be satisfied.

1. **Random Forest builds hundreds of decision trees** from your historical data
2. **Each tree asks simple questions:**
   - "Is the seat comfort rating > 3?"
   - "Is the departure delay > 30 minutes?"
   - "Did the WiFi service work well?"
3. **All trees vote** - They collectively decide: Satisfied or Not Satisfied
4. **The majority wins** - If 80% of trees say "Satisfied", the prediction is "Satisfied"

### Why Random Forest?
- ✅ **Captures patterns** in how different factors affect satisfaction
- ✅ **Handles nonlinear relationships** - A 2-hour delay doesn't mean 2x dissatisfaction
- ✅ **Resistant to overfitting** - Won't memorize training data
- ✅ **Ranks feature importance** - Shows which factors matter most

### Model Performance
- **Accuracy:** ~95% on test data (out of 100 predictions, ~95 are correct)
- **Precision:** High confidence in positive predictions
- **Recall:** Catches most dissatisfied customers
- **Training Data:** ~25,000 passenger records

---

## 🔍 SHAP Analysis - Understanding Model Decisions

### What is SHAP?

**SHAP (SHapley Additive exPlanations)** answers the question: **"WHY did the model make this prediction?"**

### Simple Analogy
Imagine a restaurant:
- Base customer satisfaction score: 50/100
- Good food adds: +15 points
- Poor service subtracts: -10 points
- Nice ambiance adds: +20 points
- **Final score: 75/100**

SHAP does the same for our model - it shows exactly how each factor contributes to the final prediction.

### What SHAP Shows in This App

#### 1. **Force Plot**
A horizontal bar chart showing how each feature pushed the prediction up or down.
```
← Pushes toward "Not Satisfied"  |  Base  |  Pushes toward "Satisfied" →
   Negative Factors              |       |     Positive Factors
```

**Example:**
```
Low seat comfort (-10) → Base prediction (50%) → High WiFi rating (+8) = Final prediction
```

#### 2. **Feature Importance**
Ranked list showing which factors had the BIGGEST impact on this specific prediction.

**Example Output:**
```
1. Seat Comfort: MOST IMPORTANT (contributed +25%)
2. Inflight WiFi: Second (contributed +15%)
3. Online Booking: Third (contributed +8%)
4. Age: Fourth (contributed +3%)
...
```

#### 3. **Prediction Confidence**
The probability score showing how confident the model is:
- **95% Confident → Satisfied** (model is very sure)
- **52% Confident → Satisfied** (model is barely sure)

### Real Example

**Input Data:**
- Seat Comfort: 5/5 (Excellent) ✅
- WiFi Service: 1/5 (Poor) ❌
- Inflight Service: 4/5 (Very Good) ✅
- Arrival Delay: 45 minutes ❌
- Cleanliness: 5/5 (Excellent) ✅

**SHAP Analysis Results:**
```
Excellent Seat Comfort        → +35% toward Satisfied
Excellent Cleanliness          → +20% toward Satisfied
Very Good Inflight Service     → +18% toward Satisfied
45-minute Arrival Delay        → -40% toward Not Satisfied
Poor WiFi Service              → -25% toward Not Satisfied
───────────────────────────────
FINAL PREDICTION: 58% → Not Satisfied (confident)
```

**What This Means:**
The model thinks the passenger will be dissatisfied because delays and WiFi issues outweigh the good service. But it's not 100% sure because other factors are positive.

---

## 💻 How to Use This Application

### Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/Singhrituraj114/AeroSatisfy-Passenger-Experience-Analytics-Prediction.git
cd AeroSatisfy-Passenger-Experience-Analytics-Prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
streamlit run app.py
```

Your browser will automatically open the app at `http://localhost:8501`

### Step 3: Make a Prediction

1. **Scroll down to the input form**
2. **Fill in passenger details:**
   - Select Gender (Male/Female)
   - Enter Age (numeric value)
   - Choose Customer Type (Loyal/Disloyal)
   - Select Travel Class (Economy/Economy Plus/Business)
   - Enter Flight Distance (miles)
   - Choose Travel Type (Business/Personal)

3. **Rate the services (0-5 scale):**
   - Move sliders for each of the 14 service categories
   - 0 = Not Applicable, 1 = Poor, 5 = Excellent

4. **Enter delays (in minutes):**
   - Departure Delay
   - Arrival Delay

5. **Click "🚀 Predict Satisfaction"**

### Step 4: View Results

The app will display:
- ✅ **Satisfaction Prediction** - Satisfied or Not Satisfied
- 📊 **Confidence Score** - How confident is the model (0-100%)
- 📈 **Top 10 Influencing Factors** - Bar chart of most important factors
- 🔍 **SHAP Analysis** - Force plot showing how factors influenced the decision
- 📋 **Detailed Probabilities** - Exact percentages for both classes

---

## 📁 Project Files Explained

```
Airline_Web_App/
├── app.py                 # Main Streamlit application
├── model.pkl              # Trained machine learning model (binary file)
├── encoders.pkl           # Data encoders for categorical features (binary)
├── features.pkl           # List of expected features (binary)
├── requirements.txt       # Python packages needed
├── runtime.txt            # Python version info
└── README.md              # Basic project info
```

### Key Files

**app.py** - The main application file
- Loads the trained model
- Creates the web interface
- Handles user input
- Makes predictions
- Generates visualizations
- Performs SHAP analysis

**model.pkl** - The trained Random Forest model
- Contains all learned patterns from 25,000+ passenger records
- Saved in binary format for fast loading
- Cannot be edited; it's the "brain" of the system

**encoders.pkl** - Data transformation rules
- Converts text inputs (like "Male"/"Female") to numbers
- Ensures consistency between training and prediction

**features.pkl** - Feature list
- List of all 22 expected input features
- Ensures the model receives data in the correct order

---

## 🔄 Data Flow: From Input to Prediction

```
┌──────────────────────────────────┐
│  1. User Enters Data             │  Example: "Gender: Male, Age: 35, WiFi: 5"
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  2. Encode Categorical Data      │  Male → 1, Female → 0
│     (Convert text to numbers)    │  Loyal → 1, Disloyal → 0
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  3. Arrange Features in Order    │  [35, 1, 5, 4, 3, ...] (22 numbers)
│     (Model expects specific      │
│      order for features)         │
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  4. Feed to Model                │  Model processes the 22 numbers
│     (Random Forest)              │  through 500+ decision trees
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  5. Get Prediction               │  Output: Satisfied (70% confidence)
│     + Probabilities              │          or Not Satisfied (30%)
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  6. Calculate SHAP Values        │  Analyze contribution of each feature
│     (Feature contributions)      │  to the prediction
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  7. Display Results              │  Show prediction, confidence,
│     + Visualizations             │  importance charts, SHAP plots
└──────────────────────────────────┘
```

---

## 📊 Example Predictions

### Prediction #1: Business Traveler

**Input:**
- Age: 45, Gender: Male, Customer Type: Loyal
- Class: Business, Distance: 2000 miles
- Seat Comfort: 5, WiFi: 5, Food: 5, Entertainment: 5
- Delays: 0 minutes

**Output:**
```
✅ SATISFIED (95% confidence)

Top Factors:
1. Excellent Seat Comfort (+25%)
2. Excellent WiFi Service (+20%)
3. Business Class (+18%)
4. No Delays (+15%)
5. Excellent Food (+12%)
```

---

### Prediction #2: Economy Traveler with Issues

**Input:**
- Age: 28, Gender: Female, Customer Type: Disloyal
- Class: Economy, Distance: 800 miles
- Seat Comfort: 2, WiFi: 1, Food: 2, Entertainment: 1
- Delays: 120 minutes

**Output:**
```
⚠️ NOT SATISFIED (88% confidence)

Top Factors:
1. Long Arrival Delay (-35%)
2. Poor WiFi Service (-20%)
3. Uncomfortable Seat (-18%)
4. Poor Entertainment (-15%)
5. Disloyal Customer Type (-10%)
```

---

## 🛠️ Technical Details for Developers

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Streamlit | Interactive web interface |
| **ML Algorithm** | scikit-learn | Machine learning model |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Explainability** | SHAP | Model interpretation |
| **Visualization** | Plotly | Interactive charts |
| **Model Saving** | joblib | Binary model storage |

### Model Training (How we trained it)

```python
# Pseudocode showing the training process
1. Load 25,000 passenger records from dataset
2. Split data:
   - 80% for training (20,000 records)
   - 20% for testing (5,000 records)
3. Encode categorical features (Gender, Class, etc.)
4. Train Random Forest:
   - 500 decision trees
   - Max depth: 10 (prevent overfitting)
   - Min samples: 5 (require evidence)
5. Evaluate on test data
6. Save model to model.pkl
```

### Model Architecture

```
Random Forest Classifier
├── Tree 1 (learns from random subset of data & features)
├── Tree 2 (learns from different random subset)
├── Tree 3
├── ...
├── Tree 500 (all independent)
└── Voting Mechanism (500 trees vote on final prediction)
```

---

## 🎓 Understanding the Output Metrics

### Confidence Score (Probability)

**What it means:**
- **90% Satisfied** = Model is 90% sure the passenger will be satisfied
- **55% Satisfied** = Model is only slightly confident (barely above 50/50)
- **51% Satisfied** = Borderline - could go either way

**Interpretation:**
- **80%+** = High confidence, reliable prediction
- **60-80%** = Medium confidence, reasonably reliable
- **50-60%** = Low confidence, use with caution

### Feature Importance

Shows how much each factor contributed to this prediction.

**Example:**
```
Seat Comfort: 35% (most important)
WiFi Service: 25%
Inflight Service: 20%
Other factors: 20%
```

**What it means:** If you change seat comfort from 5 to 1, the prediction changes the most (35% impact).

---

## 📈 Performance Metrics

### Accuracy: ~95%
Out of 100 passengers, the model correctly predicts satisfaction for 95 of them.

### False Positive Rate: Low
Rarely predicts "Satisfied" when passenger is actually dissatisfied.

### False Negative Rate: Low
Rarely misses dissatisfied customers who need help.

### Cross-validation Score: ~93%
Consistent performance across different data subsets.

---

## 🔧 Customization & Extension Ideas

### Easy Additions:
1. **Add more service categories** - Include loyalty program ratings, seat selection ease, etc.
2. **Add seasonal data** - Include month/season to capture travel patterns
3. **Add route information** - Include origin/destination to understand route-specific satisfaction
4. **Add historical data** - Include previous flight history

### Advanced Enhancements:
1. **Retrain with new data** - Include recent passenger feedback
2. **A/B Testing** - Test model changes on real passengers
3. **Integration** - Connect to airline booking system
4. **Mobile App** - Create mobile version of this web app
5. **Real-time Monitoring** - Track satisfaction trends over time

---

## ⚠️ Limitations & Important Notes

### Model Limitations:
1. **Based on historical data** - Trained on 25,000 records from 2015-2020
2. **Cannot predict individual emotions** - Satisfaction is subjective
3. **Depends on input accuracy** - Wrong ratings = wrong predictions
4. **No external factors** - Doesn't know about events, news, or special circumstances

### Data Quality:
1. Ratings must be between 0-5
2. Delays are in minutes (whole numbers)
3. Age should be realistic (0-150 years)
4. All fields should be filled (0 for "Not Applicable")

### Best Practices:
1. Use realistic service ratings
2. Enter accurate delay information
3. Don't treat this as 100% prediction - use as a guide
4. Combine with human judgment for critical decisions
5. Retrain model quarterly with new passenger data

---

## 📞 Support & Contribution

### Questions?
Open an issue on GitHub explaining:
- What you tried
- What error you got
- What you expected to happen

### Want to improve?
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Push and create a Pull Request

### Reporting Bugs:
Describe the issue with:
- Steps to reproduce
- Error message (if any)
- Expected behavior
- Screenshots (if helpful)

---

## 📜 License

This project is open-source and available for educational and commercial use.

---

## 🎯 Summary

**AeroSatisfy** is a complete solution for predicting airline passenger satisfaction:
- ✅ Takes passenger data as input
- ✅ Uses machine learning to predict satisfaction
- ✅ Explains predictions with SHAP analysis
- ✅ Provides actionable insights
- ✅ Beautiful, interactive web interface

**Perfect for:** Airlines, customer service teams, data science learning, and business intelligence.

---

**Built with ❤️ for better airline experiences**
