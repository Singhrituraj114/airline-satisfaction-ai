# 📊 Data Dictionary - Column Explanations

## Complete Guide to All Input Features

This document explains every single feature (input field) in the Airline Passenger Satisfaction Prediction model.

---

## Table of Contents
1. [Passenger Demographics](#passenger-demographics)
2. [Travel Information](#travel-information)
3. [Service Quality Ratings](#service-quality-ratings)
4. [Flight Delays](#flight-delays)
5. [Target Variable](#target-variable)
6. [Data Types & Ranges](#data-types--ranges)

---

## Passenger Demographics

### 1. **Gender**
- **Type:** Categorical (Text)
- **Possible Values:** Male, Female
- **What it means:** The biological gender of the passenger
- **Example:** Male
- **Model Converts to:** 1 (Male), 0 (Female)
- **Why it matters:** Some genders might have different satisfaction patterns based on service preferences

---

### 2. **Age**
- **Type:** Numerical (Integer)
- **Range:** 0 to 150 years
- **Realistic Range:** 1 to 100 years
- **What it means:** How old the passenger is
- **Example:** 35
- **Why it matters:** Younger and older passengers have different expectations and comfort preferences

---

### 3. **Customer Type**
- **Type:** Categorical (Text)
- **Possible Values:** 
  - Loyal Customer (flies frequently with this airline)
  - Disloyal Customer (first time or infrequent flyer)
- **What it means:** Whether this passenger is a regular customer or a one-time flyer
- **Example:** Loyal Customer
- **Model Converts to:** 1 (Loyal), 0 (Disloyal)
- **Why it matters:** Loyal customers have higher expectations; one-time flyers might be more forgiving

---

## Travel Information

### 4. **Ticket Class**
- **Type:** Categorical (Text)
- **Possible Values:** 
  - Economy (standard seats, basic service)
  - Economy Plus (slightly better seats, faster boarding)
  - Business (premium seats, full meals, priority service)
- **What it means:** Which cabin class the passenger is flying in
- **Example:** Business
- **Model Converts to:** Encoded (each class gets a number)
- **Why it matters:** Business class passengers expect better service and comfort; economy passengers have lower expectations

---

### 5. **Flight Distance**
- **Type:** Numerical (Integer)
- **Unit:** Miles
- **Range:** 0 to 5000+ miles
- **What it means:** How far the passenger is flying
- **Examples:** 
  - 200 miles = Short flight (1-2 hours)
  - 1500 miles = Medium flight (3-4 hours)
  - 3000 miles = Long flight (5+ hours)
- **Why it matters:** Longer flights require better food, entertainment, and comfort services

---

### 6. **Travel Type**
- **Type:** Categorical (Text)
- **Possible Values:** 
  - Business (work-related trip)
  - Personal (vacation or personal reasons)
- **What it means:** The purpose of the passenger's flight
- **Example:** Business
- **Model Converts to:** 1 (Business), 0 (Personal)
- **Why it matters:** Business travelers prioritize punctuality and WiFi; leisure travelers prioritize comfort and entertainment

---

## Service Quality Ratings

All service ratings use the same scale:

### Rating Scale Explanation
```
0 = Not Applicable (didn't use this service)
1 = Poor (very unsatisfied)
2 = Fair (somewhat unsatisfied)
3 = Good (neutral, acceptable)
4 = Very Good (satisfied)
5 = Excellent (very satisfied)
```

### 7. **Inflight WiFi Service**
- **What it is:** Quality of internet service during the flight
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 4 (Very Good - WiFi worked but had occasional lag)
- **Why it matters:** Business travelers rely on WiFi; entertainment for others
- **Impact on satisfaction:** HIGH ⭐⭐⭐⭐⭐

---

### 8. **Departure/Arrival Time Convenience**
- **What it is:** How convenient were the flight times?
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 3 (Good - Flight time was acceptable)
- **Why it matters:** Early morning or late night flights are inconvenient for many
- **Impact on satisfaction:** MEDIUM ⭐⭐⭐

---

### 9. **Ease of Online Booking**
- **What it is:** Was it easy to book the flight online?
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 5 (Excellent - Website was intuitive and quick)
- **Why it matters:** First impression of the airline
- **Impact on satisfaction:** MEDIUM ⭐⭐⭐

---

### 10. **Gate Location**
- **What it is:** Was the gate conveniently located at the airport?
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 2 (Fair - Had to walk far, confusing airport layout)
- **Why it matters:** Long walks at airports are tiring, especially for elderly or families
- **Impact on satisfaction:** MEDIUM ⭐⭐⭐

---

### 11. **Food and Drink**
- **What it is:** Quality of meals and beverages offered
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 3 (Good - Food was acceptable but nothing special)
- **Why it matters:** Essential on long flights; optional on short flights
- **Impact on satisfaction:** HIGH on long flights ⭐⭐⭐⭐

---

### 12. **Online Boarding**
- **What it is:** Quality of the online check-in and boarding process
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 4 (Very Good - App worked well)
- **Why it matters:** Smooth process saves time and reduces stress
- **Impact on satisfaction:** MEDIUM ⭐⭐⭐

---

### 13. **Seat Comfort**
- **What it is:** How comfortable were the seats?
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 4 (Very Good - Seats were spacious with good padding)
- **Why it matters:** CRITICAL for long flights; less important for short flights
- **Impact on satisfaction:** VERY HIGH ⭐⭐⭐⭐⭐

---

### 14. **Inflight Entertainment**
- **What it is:** Quality of movies, TV shows, music, and games available
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 3 (Good - Decent selection but outdated movies)
- **Why it matters:** Keeps passengers occupied on long flights
- **Impact on satisfaction:** MEDIUM-HIGH ⭐⭐⭐⭐

---

### 15. **On-board Service**
- **What it is:** Quality of flight attendant service (politeness, helpfulness)
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 5 (Excellent - Attendants were very attentive and friendly)
- **Why it matters:** Flight attendants directly interact with passengers
- **Impact on satisfaction:** VERY HIGH ⭐⭐⭐⭐⭐

---

### 16. **Leg Room Service**
- **What it is:** How much space is available for legs at the seat?
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 2 (Fair - Cramped, couldn't stretch legs)
- **Why it matters:** Comfort issue; affects satisfaction on flights over 3 hours
- **Impact on satisfaction:** HIGH ⭐⭐⭐⭐

---

### 17. **Baggage Handling**
- **What it is:** Quality of how luggage was handled (arrived on time, undamaged)
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 4 (Very Good - Baggage arrived quickly and intact)
- **Why it matters:** Damaged or lost baggage ruins the trip
- **Impact on satisfaction:** VERY HIGH ⭐⭐⭐⭐⭐

---

### 18. **Check-in Service**
- **What it is:** Quality of the check-in counter experience
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 3 (Good - Staff was helpful but line was long)
- **Why it matters:** First in-person interaction with airline
- **Impact on satisfaction:** MEDIUM ⭐⭐⭐

---

### 19. **Inflight Service** (Overall)
- **What it is:** Overall quality of service during flight
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 4 (Very Good - All staff were professional and helpful)
- **Why it matters:** Summary of all service interactions
- **Impact on satisfaction:** VERY HIGH ⭐⭐⭐⭐⭐

---

### 20. **Cleanliness**
- **What it is:** How clean were the aircraft, restrooms, and seats?
- **Rating:** 0 (N/A) to 5 (Excellent)
- **Example:** 5 (Excellent - Everything was spotless)
- **Why it matters:** Directly affects health and comfort perceptions
- **Impact on satisfaction:** VERY HIGH ⭐⭐⭐⭐⭐

---

## Flight Delays

### 21. **Departure Delay (minutes)**
- **Type:** Numerical (Integer)
- **Unit:** Minutes
- **Range:** 0 to 1000+ minutes
- **What it means:** How many minutes late the flight departed
- **Examples:**
  - 0 = On time
  - 15 = Left 15 minutes late
  - 120 = Left 2 hours late
- **Why it matters:** Delays cause stress and missed connections
- **Impact on satisfaction:** VERY HIGH - Delays significantly reduce satisfaction ⭐⭐⭐⭐⭐

---

### 22. **Arrival Delay (minutes)**
- **Type:** Numerical (Integer)
- **Unit:** Minutes
- **Range:** 0 to 1000+ minutes
- **What it means:** How many minutes late the flight arrived
- **Examples:**
  - 0 = Arrived on time
  - 30 = Arrived 30 minutes late
  - 180 = Arrived 3 hours late
- **Why it matters:** Late arrivals disrupt plans and cause frustration
- **Impact on satisfaction:** VERY HIGH - Strong negative impact on satisfaction ⭐⭐⭐⭐⭐

---

## Target Variable

### **Satisfaction Status** (What we predict)
- **Type:** Categorical (Binary - Two possible values)
- **Possible Values:**
  - Satisfied ✅ (passenger would recommend this airline)
  - Not Satisfied / Dissatisfied ⚠️ (passenger would NOT recommend)
- **What it means:** Is the passenger satisfied with their overall flight experience?
- **Model Output:**
  - Probability of Satisfied: 0-100%
  - Probability of Not Satisfied: 0-100%
  - Sum always equals 100%

---

## Data Types & Ranges

### Summary Table

| Feature # | Feature Name | Type | Range/Values | Required? |
|-----------|---|---|---|---|
| 1 | Gender | Categorical | Male, Female | ✅ Yes |
| 2 | Age | Numerical | 0-150 | ✅ Yes |
| 3 | Customer Type | Categorical | Loyal, Disloyal | ✅ Yes |
| 4 | Ticket Class | Categorical | Economy, Eco Plus, Business | ✅ Yes |
| 5 | Flight Distance | Numerical | 0-5000+ miles | ✅ Yes |
| 6 | Travel Type | Categorical | Business, Personal | ✅ Yes |
| 7-20 | Service Ratings (14 services) | Numerical | 0-5 | ✅ Yes |
| 21 | Departure Delay | Numerical | 0-1000+ minutes | ✅ Yes |
| 22 | Arrival Delay | Numerical | 0-1000+ minutes | ✅ Yes |

---

## 📝 Tips for Using This Data Dictionary

### When Making Predictions:
1. **Be consistent** - Use the exact values listed (e.g., "Business" not "business")
2. **Be accurate** - Wrong ratings = wrong predictions
3. **Use 0 for N/A** - If a service wasn't used, rate it 0
4. **Remember the scale** - 3 means "Good/Acceptable", not average or median

### Understanding Patterns:
- **Service ratings are most important** - They directly reflect passenger experience
- **Delays are major factors** - Even small delays reduce satisfaction
- **Seat comfort & cleanliness are critical** - These affect physical comfort
- **Class matters** - Business class customers expect more

### Common Mistakes to Avoid:
❌ Rating all services as 3 (Good) - Be honest about quality  
❌ Forgetting to enter delays - Delays heavily influence satisfaction  
❌ Using wrong category values - "Biz" instead of "Business"  
❌ Rating services that weren't used - Use 0 for not applicable  
❌ Entering unrealistic data - Age should be 1-100, not 999  

---

## 🔄 Relationship Between Features

### Features that strongly influence each other:

**Flight Class → Service Quality Expectations**
- Business class (higher expectations) → Services must be 4-5 to satisfy
- Economy class (lower expectations) → Services at 3-4 can satisfy

**Travel Distance → What Services Matter**
- Short flights (< 500 miles): Delays matter most
- Medium flights (500-2000 miles): Seat comfort + service matter
- Long flights (> 2000 miles): Comfort + entertainment matter most

**Travel Type → Priority**
- Business travel: Punctuality, WiFi, cleanliness
- Personal travel: Comfort, entertainment, food

**Delays → Overall Satisfaction**
- 0-15 minutes: Minimal impact
- 15-60 minutes: Noticeable negative impact
- 60+ minutes: Major dissatisfaction

---

## 📊 Example Data Point

**Complete example of all features:**

```
Gender: Male
Age: 45
Customer Type: Loyal Customer
Ticket Class: Business
Flight Distance: 2500 miles
Travel Type: Business

Service Ratings (0-5):
1. Inflight WiFi Service: 5 (Excellent)
2. Departure/Arrival Time Convenience: 4 (Very Good)
3. Ease of Online Booking: 5 (Excellent)
4. Gate Location: 3 (Good)
5. Food and Drink: 4 (Very Good)
6. Online Boarding: 4 (Very Good)
7. Seat Comfort: 5 (Excellent)
8. Inflight Entertainment: 4 (Very Good)
9. On-board Service: 5 (Excellent)
10. Leg Room Service: 5 (Excellent)
11. Baggage Handling: 5 (Excellent)
12. Check-in Service: 4 (Very Good)
13. Inflight Service: 5 (Excellent)
14. Cleanliness: 5 (Excellent)

Delays:
Departure Delay: 0 minutes
Arrival Delay: 5 minutes

Expected Prediction: ✅ SATISFIED (95%+ confidence)
Reason: High service ratings, no delays, business class, loyal customer
```

---

**Total Features: 22 inputs → 1 prediction (Satisfied or Not Satisfied)**
