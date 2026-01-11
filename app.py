import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Load trained artifacts
# -------------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Airline Satisfaction Predictor", page_icon="✈", layout="centered")

st.title("✈ Airline Passenger Satisfaction Predictor")
st.write("AI-powered airline customer satisfaction prediction")

st.divider()

# -------------------------------
# Passenger Info
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    cust_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])

with col2:
    age = st.number_input("Age", 0, 100, 30)
    distance = st.number_input("Flight Distance", 0, 10000, 1000)
    dep_delay = st.number_input("Departure Delay (minutes)", 0, 1000, 0)
    arr_delay = st.number_input("Arrival Delay (minutes)", 0, 1000, 0)

st.subheader("Service Ratings (0 = Very Poor, 5 = Excellent)")

wifi = st.slider("Inflight Wifi Service", 0, 5, 3)
time_conv = st.slider("Departure / Arrival Time Convenience", 0, 5, 3)
booking = st.slider("Ease of Online Booking", 0, 5, 3)
gate = st.slider("Gate Location", 0, 5, 3)
food = st.slider("Food and Drink", 0, 5, 3)
online_boarding = st.slider("Online Boarding", 0, 5, 3)
seat = st.slider("Seat Comfort", 0, 5, 3)
entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
onboard = st.slider("On-board Service", 0, 5, 3)
legroom = st.slider("Leg Room Service", 0, 5, 3)
baggage = st.slider("Baggage Handling", 0, 5, 3)
checkin = st.slider("Check-in Service", 0, 5, 3)
inflight = st.slider("Inflight Service", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🚀 Predict Satisfaction", use_container_width=True):

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

    # Encode categorical
    for col in encoders:
        input_data[col] = encoders[col].transform([input_data[col]])[0]

    # Convert to DataFrame
    X = pd.DataFrame([input_data])
    X = X[features]

    # Predict
    proba = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]
    confidence = proba[prediction] * 100

    st.divider()

    if prediction == 1:
        st.success(f"Passenger is likely to be **SATISFIED**")
    else:
        st.error(f"Passenger is likely to be **NOT SATISFIED**")

    st.write(f"Confidence: **{confidence:.1f}%**")
    st.progress(int(confidence))

   # -------------------------------
# Explainable AI (Visual + Interactive)
# -------------------------------
st.subheader("🧠 Why this prediction was made")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

# Normalize for nicer bars
importance_df["Importance %"] = (importance_df["Importance"] / importance_df["Importance"].sum()) * 100
importance_df = importance_df.sort_values("Importance %", ascending=False)

top = importance_df.head(10)

st.write("Top factors influencing this prediction")

# Progress-bar style importance
for _, row in top.iterrows():
    st.markdown(f"**{row['Feature']}**")
    st.progress(int(row["Importance %"]))
    st.write(f"{row['Importance %']:.1f}% impact")
    st.markdown("---")

# Graph version
st.subheader("📊 Visual Importance Chart")

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top["Feature"], top["Importance %"])
ax.invert_yaxis()
ax.set_xlabel("Influence (%)")
ax.set_title("Key factors affecting passenger satisfaction")

st.pyplot(fig)

