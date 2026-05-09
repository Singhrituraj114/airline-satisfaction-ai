import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD TRAINED MODEL AND ARTIFACTS
# ============================================================================
@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model, encoders, and features"""
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    features = joblib.load("features.pkl")
    return model, encoders, features

model, encoders, features = load_model_artifacts()

# Create SHAP explainer without caching (caching causes pickling issues)
try:
    shap_explainer = shap.TreeExplainer(model)
except Exception as e:
    shap_explainer = None

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="✈️ Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM 3D CSS STYLING
# ============================================================================
custom_css = """
    <style>
    /* Root colors - Beautiful gradient palette */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #4facfe 100%);
        --success-gradient: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        --danger-gradient: linear-gradient(135deg, #ff6b6b 0%, #ff0000 100%);
    }

    /* Main background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }

    /* Remove default background */
    .main {
        background: transparent !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        box-shadow: -10px 0 30px rgba(0, 0, 0, 0.5);
    }

    /* Section container - 3D card effect */
    .section-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(240, 147, 251, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 
            0 20px 60px rgba(102, 126, 234, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .section-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 30px 80px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }

    /* Input elements styling */
    .stSelectbox, .stNumberInput, .stSlider {
        background: rgba(255, 255, 255, 0.05) !important;
    }

    .stSelectbox [data-testid="stSelectbox"] > div,
    .stNumberInput [data-testid="stNumberInput"] > div {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(240, 147, 251, 0.2));
        border: 2px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 10px;
    }

    /* Button styling - 3D effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 18px 40px;
        font-weight: 700;
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 
            0 15px 35px rgba(102, 126, 234, 0.4),
            0 0 0 0 rgba(102, 126, 234, 0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 25px 45px rgba(102, 126, 234, 0.6),
            0 0 20px rgba(240, 147, 251, 0.4);
    }

    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 
            0 10px 20px rgba(102, 126, 234, 0.4),
            0 0 10px rgba(240, 147, 251, 0.2);
    }

    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }

    /* Success/Error badges */
    .success-badge {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0, 212, 255, 0.4);
        border: 2px solid rgba(0, 212, 255, 0.5);
        backdrop-filter: blur(10px);
    }

    .error-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff0000 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(255, 107, 107, 0.4);
        border: 2px solid rgba(255, 107, 107, 0.5);
        backdrop-filter: blur(10px);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(79, 172, 254, 0.15) 100%);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 15px 40px rgba(0, 212, 255, 0.25);
        backdrop-filter: blur(10px);
    }

    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #f093fb, transparent);
        margin: 30px 0;
    }

    /* Subheader styling */
    .subheader-text {
        color: #ffffff;
        font-size: 20px;
        font-weight: 600;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(240, 147, 251, 0.2));
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        border-color: #f093fb;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(240, 147, 251, 0.2)) !important;
        border-radius: 12px;
    }

    /* General text */
    .css-12ttj6m {
        color: #ffffff;
    }

    /* Links */
    a {
        color: #4facfe !important;
        text-decoration: none;
    }

    a:hover {
        color: #f093fb !important;
        text-decoration: underline;
    }

    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7);
        }
        50% {
            box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
        }
    }

    .pulse {
        animation: pulse 2s infinite;
    }
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div style='text-align: center; padding: 40px 0; margin-bottom: 40px;'>
        <h1 style='font-size: 48px; margin: 0; text-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);'>
            ✈️ AIRLINE SATISFACTION PREDICTOR
        </h1>
        <p style='font-size: 18px; color: #b0b0ff; margin: 10px 0 0 0; text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);'>
            🚀 Advanced AI-Powered Prediction System
        </p>
        <p style='font-size: 14px; color: #8b8bff; margin: 5px 0; letter-spacing: 2px;'>
            Powered by Machine Learning | Real-Time Analysis
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); border-radius: 2px; margin: 20px 0;'></div>
""", unsafe_allow_html=True)

# ============================================================================
# INPUT FORM - PASSENGER INFO
# ============================================================================
st.markdown("<h2 style='text-align: center; margin-top: 40px;'>📝 Enter Passenger Details</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class='section-card'>
            <div style='font-size: 18px; font-weight: 700; color: #f093fb; margin-bottom: 15px;'>
                👤 Passenger Information
            </div>
        </div>
    """, unsafe_allow_html=True)
    gender = st.selectbox("👥 Gender", ["Male", "Female"])
    cust_type = st.selectbox("🎯 Customer Type", ["Loyal Customer", "disloyal Customer"])
    age = st.number_input("🎂 Age", min_value=0, max_value=100, value=30)
    travel_type = st.selectbox("📌 Type of Travel", ["Business travel", "Personal Travel"])

with col2:
    st.markdown("""
        <div class='section-card'>
            <div style='font-size: 18px; font-weight: 700; color: #667eea; margin-bottom: 15px;'>
                ✈️ Flight Details
            </div>
        </div>
    """, unsafe_allow_html=True)
    travel_class = st.selectbox("💺 Class", ["Eco", "Eco Plus", "Business"])
    distance = st.number_input("🌍 Flight Distance (km)", min_value=0, value=1000)
    dep_delay = st.number_input("⏱️ Departure Delay (min)", min_value=0, value=0)
    arr_delay = st.number_input("⌛ Arrival Delay (min)", min_value=0, value=0)

with col3:
    st.markdown("""
        <div class='section-card'>
            <div style='font-size: 18px; font-weight: 700; color: #4facfe; margin-bottom: 15px;'>
                ⭐ Quick Ratings
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("*0 = Very Poor | 5 = Excellent*")
    wifi = st.slider("📡 WiFi", 0, 5, 3)
    food = st.slider("🍽️ Food & Drink", 0, 5, 3)
    seat = st.slider("💺 Seat Comfort", 0, 5, 3)
    entertainment = st.slider("🎬 Entertainment", 0, 5, 3)

st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); border-radius: 2px; margin: 30px 0;'></div>", unsafe_allow_html=True)

# ============================================================================
# SERVICE RATINGS INPUT (0-5 SCALE)
# ============================================================================
st.markdown("<h2 style='text-align: center;'>⭐ Additional Service Ratings</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0b0ff; font-size: 14px;'>Fine-tune your ratings (0-5 scale)</p>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='section-card'><b>🕐 Timing</b></div>", unsafe_allow_html=True)
    time_conv = st.slider("Departure/Arrival Time", 0, 5, 3, key="time")
    booking = st.slider("Online Booking", 0, 5, 3, key="book")
    gate = st.slider("Gate Location", 0, 5, 3, key="gate")

with col2:
    st.markdown("<div class='section-card'><b>🍴 Services</b></div>", unsafe_allow_html=True)
    online_boarding = st.slider("Online Boarding", 0, 5, 3, key="board")
    onboard = st.slider("On-board Service", 0, 5, 3, key="onboard")
    legroom = st.slider("Leg Room", 0, 5, 3, key="leg")

with col3:
    st.markdown("<div class='section-card'><b>📦 Logistics</b></div>", unsafe_allow_html=True)
    baggage = st.slider("Baggage Handling", 0, 5, 3, key="bag")
    checkin = st.slider("Check-in Service", 0, 5, 3, key="check")
    inflight = st.slider("Inflight Service", 0, 5, 3, key="inf")

with col4:
    st.markdown("<div class='section-card'><b>✨ Comfort</b></div>", unsafe_allow_html=True)
    cleanliness = st.slider("Cleanliness", 0, 5, 3, key="clean")

st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); border-radius: 2px; margin: 30px 0;'></div>", unsafe_allow_html=True)

# ============================================================================
# PREDICTION BUTTON AND LOGIC
# ============================================================================
st.markdown("<h2 style='text-align: center;'>🔮 Ready for Prediction?</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 PREDICT SATISFACTION NOW", use_container_width=True, key="predict"):
        
        # Prepare input data
        input_data = {
            "Gender": gender,
            "Customer Type": cust_type,
            "Type of Travel": travel_type,
            "Class": travel_class,
            "Age": age,
            "Flight Distance": distance,
            "Inflight wifi service": wifi,
            "Departure/Arrival time convenient": time_conv,
            "Ease of Online booking": booking,
            "Gate location": gate,
            "Food and drink": food,
            "Online boarding": online_boarding,
            "Seat comfort": seat,
            "Inflight entertainment": entertainment,
            "On-board service": onboard,
            "Leg room service": legroom,
            "Baggage handling": baggage,
            "Checkin service": checkin,
            "Inflight service": inflight,
            "Cleanliness": cleanliness,
            "Departure Delay in Minutes": dep_delay,
            "Arrival Delay in Minutes": arr_delay
        }
        
        # Encode categorical features
        for col in encoders:
            input_data[col] = encoders[col].transform([input_data[col]])[0]
        
        # Convert to DataFrame with correct feature order
        X = pd.DataFrame([input_data])[features]
        
        # Make prediction
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = proba[prediction] * 100
        
        st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); border-radius: 2px; margin: 30px 0;'></div>", unsafe_allow_html=True)
        
        # ========================================================================
        # DISPLAY PREDICTION RESULT
        # ========================================================================
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.markdown("""
                    <div class='success-badge'>
                        <h2 style='margin: 0; font-size: 36px; -webkit-text-fill-color: white; background: none;'>✅ SATISFIED</h2>
                        <p style='margin: 10px 0 0 0; font-size: 18px; color: white;'>Passenger is likely to be satisfied!</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='error-badge'>
                        <h2 style='margin: 0; font-size: 36px; -webkit-text-fill-color: white; background: none;'>⚠️ NOT SATISFIED</h2>
                        <p style='margin: 10px 0 0 0; font-size: 18px; color: white;'>Passenger needs attention</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); border-radius: 2px; margin: 30px 0;'></div>", unsafe_allow_html=True)
        
        # Confidence with gauge
        st.markdown(f"<h3 style='text-align: center; font-size: 28px;'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)
        
        fig = go.Figure(data=[
            go.Indicator(
                mode="gauge+number+delta",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Prediction Confidence"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255, 107, 107, 0.3)"},
                        {'range': [50, 100], 'color': "rgba(0, 212, 255, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            )
        ])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=14),
            height=400,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); border-radius: 2px; margin: 30px 0;'></div>", unsafe_allow_html=True)
        
        # ========================================================================
        # BEAUTIFUL FEATURE IMPACT ANALYSIS SECTION
        # ========================================================================
        st.markdown("""
            <div style='text-align: center; margin: 40px 0 30px 0;'>
                <h1 style='background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 42px; margin: 0;'>
                    🔍 Explainability Analysis
                </h1>
                <p style='color: #a0aec0; font-size: 14px; margin-top: 10px;'>Deep dive into what influenced this prediction</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); border-radius: 2px;'></div>", unsafe_allow_html=True)
        
        if shap_explainer is not None:
            try:
                # Calculate feature contributions by checking impact of each feature
                original_proba = model.predict_proba(X)[0][1]  # probability of satisfied
                
                contributions = []
                
                # For each feature, calculate how much it contributes to the prediction
                for i in range(len(features)):
                    X_baseline = X.copy()
                    X_baseline.iloc[0, i] = 0
                    baseline_proba = model.predict_proba(X_baseline)[0][1]
                    contribution = abs(original_proba - baseline_proba)
                    contributions.append(contribution)
                
                # Create DataFrame with contributions
                shap_importance_df = pd.DataFrame({
                    "Feature": features,
                    "Impact": contributions
                }).sort_values(by="Impact", ascending=False)
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Impact Chart", "📈 Top 3 Factors", "📋 Full Breakdown"])
                
                with tab1:
                    # Horizontal bar chart - beautiful version
                    top_10 = shap_importance_df.head(10)
                    
                    fig = px.bar(
                        top_10,
                        x="Impact",
                        y="Feature",
                        orientation='h',
                        title="Top 10 Features by Impact Strength",
                        labels={"Impact": "Impact Strength", "Feature": "Feature Name"},
                        color="Impact",
                        color_continuous_scale="Rainbow"
                    )
                    fig.update_traces(marker_line_width=2, marker_line_color="rgba(255,255,255,0.3)")
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.05)",
                        font=dict(color="#ffffff", size=12),
                        height=500,
                        showlegend=False,
                        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", title_font=dict(size=14)),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", title_font=dict(size=14)),
                        hovermode="y unified",
                        margin=dict(l=200, r=100, t=100, b=50)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Top 3 most impactful factors with detailed cards
                    st.markdown("<h3 style='color: #667eea; text-align: center;'>🏆 Your Top 3 Influencing Factors</h3>", unsafe_allow_html=True)
                    
                    cols = st.columns(3)
                    for idx, (col, (_, row)) in enumerate(zip(cols, shap_importance_df.head(3).iterrows())):
                        with col:
                            impact_pct = (row['Impact'] / shap_importance_df['Impact'].max()) * 100
                            medal = ["🥇", "🥈", "🥉"][idx]
                            
                            st.markdown(f"""
                                <div style='
                                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(240, 147, 251, 0.2));
                                    border: 2px solid rgba(102, 126, 234, 0.5);
                                    border-radius: 15px;
                                    padding: 20px;
                                    text-align: center;
                                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                                    backdrop-filter: blur(10px);
                                '>
                                    <h2 style='margin: 0; color: #667eea;'>{medal}</h2>
                                    <p style='margin: 10px 0 0 0; color: #e2e8f0; font-size: 14px; font-weight: bold;'>{row['Feature']}</p>
                                    <div style='
                                        background: linear-gradient(90deg, #667eea, #f093fb);
                                        height: 6px;
                                        border-radius: 3px;
                                        margin: 15px 0;
                                    '></div>
                                    <p style='margin: 0; color: #a0aec0; font-size: 12px;'>Impact: {impact_pct:.1f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                
                with tab3:
                    # Detailed breakdown table
                    st.markdown("<h3 style='color: #667eea; margin-bottom: 20px;'>📊 Complete Feature Impact Breakdown</h3>", unsafe_allow_html=True)
                    
                    display_df = shap_importance_df.copy()
                    display_df['Impact %'] = (display_df['Impact'] / display_df['Impact'].max() * 100).round(2)
                    display_df['Impact Score'] = display_df['Impact'].round(4)
                    display_df = display_df[['Feature', 'Impact Score', 'Impact %']].reset_index(drop=True)
                    display_df.index = display_df.index + 1
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "Impact Score": st.column_config.NumberColumn(format="%.4f"),
                            "Impact %": st.column_config.NumberColumn(format="%.2f%%")
                        }
                    )
                    
                    # Download button for detailed analysis
                    csv = display_df.to_csv(index=True)
                    st.download_button(
                        label="📥 Download Impact Analysis",
                        data=csv,
                        file_name="feature_impact_analysis.csv",
                        mime="text/csv"
                    )
                
                # Beautiful insight box
                st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); border-radius: 2px; margin: 30px 0;'></div>", unsafe_allow_html=True)
                
                top_factor = shap_importance_df.iloc[0]
                second_factor = shap_importance_df.iloc[1]
                
                st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(79, 172, 254, 0.1));
                        border-left: 4px solid #667eea;
                        border-radius: 8px;
                        padding: 20px;
                        margin: 20px 0;
                    '>
                        <p style='color: #e2e8f0; font-size: 16px; margin: 0; line-height: 1.6;'>
                            <strong>💡 Key Insight:</strong> <span style='color: #667eea;'>{top_factor['Feature']}</span> 
                            is the <strong>primary driver</strong> of this prediction, followed by 
                            <span style='color: #764ba2;'>{second_factor['Feature']}</span>. 
                            These two factors have the highest impact on whether the passenger is satisfied or not.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
                st.info("Showing global feature importance instead...")
                
                # Fallback to static importance
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False).head(10)
                
                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    title="Top 10 Features Affecting Satisfaction Prediction",
                    labels={"Importance": "Importance Score", "Feature": "Features"},
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                    height=400,
                    showlegend=False,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    hovermode="y unified"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to static importance if explainer not available
            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)
            
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation='h',
                title="Top 10 Features Affecting Satisfaction Prediction",
                labels={"Importance": "Importance Score", "Feature": "Features"},
                color="Importance",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
                height=400,
                showlegend=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                hovermode="y unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #4facfe); border-radius: 2px; margin: 30px 0;'></div>", unsafe_allow_html=True)
        
        # ========================================================================
        # DETAILED PROBABILITIES
        # ========================================================================
        st.markdown("<h2 style='text-align: center;'>📊 Detailed Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <div style='font-weight: 700; color: #667eea; margin-bottom: 15px; font-size: 18px;'>
                        📈 Prediction Probabilities
                    </div>
            """, unsafe_allow_html=True)
            st.write(f"✅ **Satisfied**: {proba[1]*100:.2f}%")
            st.write(f"⚠️ **Not Satisfied**: {proba[0]*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <div style='font-weight: 700; color: #4facfe; margin-bottom: 15px; font-size: 18px;'>
                        ✈️ Passenger Summary
                    </div>
            """, unsafe_allow_html=True)
            st.write(f"💺 **Class**: {travel_class}")
            st.write(f"🌍 **Distance**: {distance} km")
            st.write(f"⏱️ **Total Delays**: {dep_delay + arr_delay} minutes")
            st.markdown("</div>", unsafe_allow_html=True)
