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
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="✈️ Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    features = joblib.load("features.pkl")
    return model, encoders, features

model, encoders, features = load_model_artifacts()

try:
    shap_explainer = shap.TreeExplainer(model)
except:
    shap_explainer = None

# ============================================================================
# WORLD-CLASS STYLING
# ============================================================================
st.markdown("""
    <style>
    /* Global Styles */
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Main container */
    .main { 
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* Header Styles */
    .header-main {
        text-align: center;
        padding: 40px 20px 50px 20px;
        background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(240,147,251,0.1));
        border-radius: 20px;
        border: 2px solid rgba(102,126,234,0.2);
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        animation: slideDown 0.6s ease-out;
    }
    
    .header-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 48px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .header-subtitle {
        color: #a0aec0;
        font-size: 16px;
        margin-top: 10px;
        font-weight: 300;
    }
    
    /* Section Cards */
    .section-header {
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(79,172,254,0.15));
        border-left: 4px solid #667eea;
        padding: 20px 25px;
        border-radius: 12px;
        margin: 30px 0 20px 0;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .section-title {
        background: linear-gradient(90deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    
    /* Input Cards */
    .input-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(240,147,251,0.08));
        border: 1.5px solid rgba(102,126,234,0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 12px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .input-card:hover {
        border-color: rgba(102,126,234,0.6);
        box-shadow: 0 8px 25px rgba(102,126,234,0.2);
        transform: translateY(-2px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(79,172,254,0.12));
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102,126,234,0.2);
        border-color: rgba(102,126,234,0.6);
    }
    
    .metric-value {
        background: linear-gradient(90deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 36px;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result Badges */
    .result-success {
        background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,153,255,0.15));
        border: 2px solid rgba(0,212,255,0.4);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(0,212,255,0.1);
    }
    
    .result-danger {
        background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,0,0,0.15));
        border: 2px solid rgba(255,107,107,0.4);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(255,107,107,0.1);
    }
    
    .result-title {
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 10px 0;
        letter-spacing: 1px;
    }
    
    .result-subtitle {
        color: #a0aec0;
        font-size: 16px;
        margin: 0;
    }
    
    /* Gradient Divider */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 2px;
        margin: 30px 0;
    }
    
    /* Factor Cards */
    .factor-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(240,147,251,0.1));
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .factor-card:hover {
        transform: translateX(8px);
        border-color: rgba(102,126,234,0.6);
        box-shadow: 0 8px 20px rgba(102,126,234,0.15);
    }
    
    /* Animations */
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(15,15,35,0.9), rgba(26,26,62,0.9)) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(102,126,234,0.4) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, rgba(102,126,234,0.1), rgba(79,172,254,0.1));
        border-radius: 10px;
        padding: 5px;
        border: 1px solid rgba(102,126,234,0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 8px !important;
    }
    
    /* Slider styling */
    .stSlider > label {
        color: #a0aec0 !important;
        font-weight: 600 !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div class='header-main'>
        <h1 class='header-title'>✈️ AIRLINE PASSENGER SATISFACTION</h1>
        <p class='header-subtitle'>Advanced AI-Powered Prediction System • Real-Time Analytics</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# INTRODUCTION SECTION
# ============================================================================
st.markdown("""
    <div class='section-header'>
        <span style='font-size: 24px;'>📋</span>
        <h2 class='section-title'>Input Passenger Information</h2>
    </div>
""", unsafe_allow_html=True)

# Create organized input sections
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='input-card'><b>👤 Passenger Profile</b></div>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    age = st.slider("Age", 18, 80, 35, key="age")
    cust_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"], key="cust")

with col2:
    st.markdown("<div class='input-card'><b>✈️ Flight Details</b></div>", unsafe_allow_html=True)
    travel_type = st.selectbox("Type of Travel", ["Personal Travel", "Business travel"], key="travel")
    travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"], key="class")
    distance = st.slider("Flight Distance (km)", 100, 5000, 1500, key="dist")

with col3:
    st.markdown("<div class='input-card'><b>⏱️ Delay Information</b></div>", unsafe_allow_html=True)
    dep_delay = st.slider("Departure Delay (min)", 0, 1500, 10, key="dep_delay")
    arr_delay = st.slider("Arrival Delay (min)", 0, 1500, 10, key="arr_delay")

# Service Ratings Section
st.markdown("""
    <div class='section-header'>
        <span style='font-size: 24px;'>⭐</span>
        <h2 class='section-title'>Service Ratings (0-5 Scale)</h2>
    </div>
""", unsafe_allow_html=True)

# Grid of service ratings
col1, col2, col3, col4 = st.columns(4)

with col1:
    wifi = st.slider("WiFi Service", 0, 5, 3, key="wifi")
    time_conv = st.slider("Departure/Arrival Time", 0, 5, 3, key="time")
    booking = st.slider("Online Booking", 0, 5, 3, key="book")
    gate = st.slider("Gate Location", 0, 5, 3, key="gate")

with col2:
    food = st.slider("Food & Drink", 0, 5, 3, key="food")
    online_boarding = st.slider("Online Boarding", 0, 5, 3, key="board")
    onboard = st.slider("On-board Service", 0, 5, 3, key="onboard")
    legroom = st.slider("Leg Room", 0, 5, 3, key="leg")

with col3:
    baggage = st.slider("Baggage Handling", 0, 5, 3, key="bag")
    checkin = st.slider("Check-in Service", 0, 5, 3, key="check")
    inflight = st.slider("Inflight Service", 0, 5, 3, key="inf")
    seat = st.slider("Seat Comfort", 0, 5, 3, key="seat")

with col4:
    entertainment = st.slider("Entertainment", 0, 5, 3, key="ent")
    cleanliness = st.slider("Cleanliness", 0, 5, 3, key="clean")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================================
# PREDICTION BUTTON
# ============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🚀 PREDICT SATISFACTION", use_container_width=True, key="predict")

if predict_btn:
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
    
    # Convert to DataFrame
    X = pd.DataFrame([input_data])[features]
    
    # Make prediction
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = proba[prediction] * 100
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ========================================================================
    # PREDICTION RESULT
    # ========================================================================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction == 1:
            st.markdown("""
                <div class='result-success'>
                    <h2 class='result-title'>✅ SATISFIED</h2>
                    <p class='result-subtitle'>Passenger is likely to be satisfied!</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='result-danger'>
                    <h2 class='result-title'>⚠️ NOT SATISFIED</h2>
                    <p class='result-subtitle'>Passenger needs attention</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ========================================================================
    # CONFIDENCE GAUGE
    # ========================================================================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class='section-header'>
                <span style='font-size: 20px;'>📊</span>
                <h3 class='section-title'>Prediction Confidence</h3>
            </div>
        """, unsafe_allow_html=True)
        
        fig_gauge = go.Figure(data=[
            go.Indicator(
                mode="gauge+number+delta",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Confidence Score"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 25], 'color': "rgba(255,107,107,0.2)"},
                        {'range': [25, 50], 'color': "rgba(255,170,0,0.2)"},
                        {'range': [50, 75], 'color': "rgba(0,212,255,0.2)"},
                        {'range': [75, 100], 'color': "rgba(0,255,170,0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            )
        ])
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff", size=14),
            height=350,
            margin=dict(l=0, r=0, t=60, b=0)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # ========================================================================
    # PROBABILITY BREAKDOWN
    # ========================================================================
    st.markdown("""
        <div class='section-header'>
            <span style='font-size: 20px;'>📈</span>
            <h3 class='section-title'>Probability Analysis</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Satisfied Probability</div>
                <div class='metric-value'>{proba[1]*100:.1f}%</div>
                <div style='margin-top: 10px; height: 6px; background: linear-gradient(90deg, #667eea, #00ffff); border-radius: 3px;'></div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Not Satisfied Probability</div>
                <div class='metric-value'>{proba[0]*100:.1f}%</div>
                <div style='margin-top: 10px; height: 6px; background: linear-gradient(90deg, #ff6b6b, #ff0000); border-radius: 3px;'></div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Overall Confidence</div>
                <div class='metric-value'>{confidence:.1f}%</div>
                <div style='margin-top: 10px; height: 6px; background: linear-gradient(90deg, #667eea, #f093fb); border-radius: 3px;'></div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ========================================================================
    # EXPLAINABILITY ANALYSIS
    # ========================================================================
    st.markdown("""
        <div class='section-header'>
            <span style='font-size: 24px;'>🔍</span>
            <h2 class='section-title'>Feature Impact Analysis</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if shap_explainer is not None:
        try:
            # Calculate contributions
            original_proba = model.predict_proba(X)[0][1]
            contributions = []
            
            for i in range(len(features)):
                X_baseline = X.copy()
                X_baseline.iloc[0, i] = 0
                baseline_proba = model.predict_proba(X_baseline)[0][1]
                contribution = abs(original_proba - baseline_proba)
                contributions.append(contribution)
            
            shap_importance_df = pd.DataFrame({
                "Feature": features,
                "Impact": contributions
            }).sort_values(by="Impact", ascending=False)
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Impact Chart", "🏆 Top 3 Factors", "📋 Full Breakdown"])
            
            with tab1:
                top_10 = shap_importance_df.head(10)
                
                fig = px.bar(
                    top_10,
                    x="Impact",
                    y="Feature",
                    orientation='h',
                    title="Top 10 Features by Impact Strength",
                    color="Impact",
                    color_continuous_scale="Spectral"
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
                st.markdown("<h3 style='color: #667eea; text-align: center;'>🏆 Your Top 3 Influencing Factors</h3>", unsafe_allow_html=True)
                
                cols = st.columns(3)
                for idx, (col, (_, row)) in enumerate(zip(cols, shap_importance_df.head(3).iterrows())):
                    with col:
                        impact_pct = (row['Impact'] / shap_importance_df['Impact'].max()) * 100
                        medal = ["🥇", "🥈", "🥉"][idx]
                        
                        st.markdown(f"""
                            <div class='factor-card' style='text-align: center; padding: 25px !important;'>
                                <h2 style='margin: 0; color: #667eea; font-size: 28px;'>{medal}</h2>
                                <p style='margin: 12px 0 0 0; color: #e2e8f0; font-size: 14px; font-weight: bold;'>{row['Feature']}</p>
                                <div style='background: linear-gradient(90deg, #667eea, #f093fb); height: 6px; border-radius: 3px; margin: 15px 0;'></div>
                                <p style='margin: 0; color: #a0aec0; font-size: 12px;'>Impact: <strong>{impact_pct:.1f}%</strong></p>
                            </div>
                        """, unsafe_allow_html=True)
            
            with tab3:
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
                
                csv = display_df.to_csv(index=True)
                st.download_button(
                    label="📥 Download Analysis",
                    data=csv,
                    file_name="feature_impact_analysis.csv",
                    mime="text/csv"
                )
            
            # Insights
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            top_factor = shap_importance_df.iloc[0]
            second_factor = shap_importance_df.iloc[1]
            
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(79,172,254,0.15));
                    border-left: 4px solid #667eea;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                '>
                    <p style='color: #e2e8f0; font-size: 16px; margin: 0; line-height: 1.6;'>
                        <strong>💡 Key Insight:</strong> <span style='color: #667eea; font-weight: bold;'>{top_factor['Feature']}</span> 
                        is the <strong>primary driver</strong> of this prediction, followed by 
                        <span style='color: #764ba2; font-weight: bold;'>{second_factor['Feature']}</span>. 
                        These two factors have the highest impact on the satisfaction outcome.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Analysis Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 30px; color: #a0aec0; font-size: 12px;'>
        <p style='margin: 0;'>🚀 Powered by Advanced Machine Learning • Real-time Predictions</p>
        <p style='margin: 5px 0 0 0;'>© 2026 Airline Passenger Satisfaction Predictor</p>
    </div>
""", unsafe_allow_html=True)
