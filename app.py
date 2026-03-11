

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Attrition Predictor", layout="centered")

# 1. Load Data and Train Model (Cached so it only runs once when the app starts)
@st.cache_resource
def train_model():
    # Load your dataset
    df = pd.read_csv('employee_performance_workload_attrition.csv')
    
    # Setup encoders and map categorical data
    le_dept = LabelEncoder()
    le_role = LabelEncoder()
    df['department'] = le_dept.fit_transform(df['department'])
    df['role_level'] = le_role.fit_transform(df['role_level'])
    df['attrition'] = df['attrition'].map({'Yes': 1, 'No': 0})
    
    # Apply your engineered features
    df['projects_per_hour'] = df['projects_handled'] / df['avg_weekly_hours']
    df["absence_rate"] = df["absences_days"] / 365
    df["performance_efficiency"] = df["performance_rating"] / df["avg_weekly_hours"]
    df["stress_index"] = df["avg_weekly_hours"] * df["absences_days"]
    df["satisfaction_score"] = df["job_satisfaction"] * df["performance_rating"]
    
    # Define features and target (dropping employee_id)
    X = df.drop(columns=['employee_id', 'attrition'])
    y = df['attrition']
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, le_dept, le_role, X.columns

# Load the model and tools into memory
model, scaler, le_dept, le_role, feature_cols = train_model()

# 2. Build the Web App Interface
st.title("🚀 Employee Attrition Predictor")
st.write("Enter the employee's details below to predict their likelihood of leaving the company.")

# Create a two-column layout for a clean design
col1, col2 = st.columns(2)

with col1:
    # Use the encoders to automatically populate the dropdown options with the real text names
    dept = st.selectbox("Department", le_dept.classes_)
    role = st.selectbox("Role Level", le_role.classes_)
    salary = st.number_input("Monthly Salary ($)", min_value=10000, max_value=250000, value=45000, step=1000)
    hours = st.slider("Avg Weekly Hours", 20, 100, 80)

with col2:
    projects = st.slider("Projects Handled", 1, 10, 4)
    perf = st.slider("Performance Rating", 1, 5, 5)
    absences = st.slider("Absence Days", 0, 30, 0)
    sat = st.slider("Job Satisfaction", 1, 5, 1)

# 3. Prediction Button Logic
if st.button("Predict Attrition", type="primary"):
    # Create a DataFrame for the single new employee
    new_data = pd.DataFrame([{
        "department": le_dept.transform([dept])[0],
        "role_level": le_role.transform([role])[0],
        "monthly_salary": salary,
        "avg_weekly_hours": hours,
        "projects_handled": projects,
        "performance_rating": perf,
        "absences_days": absences,
        "job_satisfaction": sat
    }])
    
    # Calculate the engineered features for the new employee
    new_data['projects_per_hour'] = new_data['projects_handled'] / new_data['avg_weekly_hours']
    new_data["absence_rate"] = new_data["absences_days"] / 365
    new_data["performance_efficiency"] = new_data["performance_rating"] / new_data["avg_weekly_hours"]
    new_data["stress_index"] = new_data["avg_weekly_hours"] * new_data["absences_days"]
    new_data["satisfaction_score"] = new_data["job_satisfaction"] * new_data["performance_rating"]
    
    # Ensure column order perfectly matches the training data
    new_data = new_data[feature_cols]
    
    # Scale the data and generate predictions
    scaled_data = scaler.transform(new_data)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    # Display the result
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **YES (Employee is likely to leave)** - Attrition Probability: {probability:.1%}")
    else:
        st.success(f"✅ **NO (Employee is likely to stay)** - Attrition Probability: {probability:.1%}")