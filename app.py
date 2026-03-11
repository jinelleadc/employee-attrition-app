import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import plotly.graph_objects as go

# Use a wide layout so tabs and charts have room to breathe
st.set_page_config(page_title="Attrition Predictor", layout="wide")

# 1. Load Data and Train Model
@st.cache_resource
def train_model():
    # Load raw data (keeping a copy for the EDA tab)
    raw_df = pd.read_csv('employee_performance_workload_attrition.csv')
    df = raw_df.copy()
    
    le_dept = LabelEncoder()
    le_role = LabelEncoder()
    df['department'] = le_dept.fit_transform(df['department'])
    df['role_level'] = le_role.fit_transform(df['role_level'])
    df['attrition'] = df['attrition'].map({'Yes': 1, 'No': 0})
    
    # Apply engineered features
    df['projects_per_hour'] = df['projects_handled'] / df['avg_weekly_hours']
    df["absence_rate"] = df["absences_days"] / 365
    df["performance_efficiency"] = df["performance_rating"] / df["avg_weekly_hours"]
    df["stress_index"] = df["avg_weekly_hours"] * df["absences_days"]
    df["satisfaction_score"] = df["job_satisfaction"] * df["performance_rating"]
    
    X = df.drop(columns=['employee_id', 'attrition'])
    y = df['attrition']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, le_dept, le_role, X.columns, raw_df

model, scaler, le_dept, le_role, feature_cols, raw_df = train_model()

# 2. Main Title
st.title("Employee Attrition Predictor")
st.markdown("Analyze workforce data and predict employee turnover using Machine Learning.")

# 3. Create Tabs
tab1, tab2, tab3 = st.tabs(["Predict Attrition", "Exploratory Data Analysis", "Global Model Insights"])

# ==========================================
# TAB 1: PREDICTION INTERFACE
# ==========================================
with tab1:
    st.subheader("New Employee Data Entry")
    col1, col2 = st.columns(2)

    with col1:
        dept = st.selectbox("Department", le_dept.classes_)
        role = st.selectbox("Role Level", le_role.classes_)
        salary = st.number_input("Monthly Salary ($)", min_value=10000, value=45000, step=1000)
        hours = st.slider("Avg Weekly Hours", 20, 100, 80)

    with col2:
        projects = st.slider("Projects Handled", 1, 10, 4)
        perf = st.slider("Performance Rating", 1, 5, 5)
        absences = st.slider("Absence Days", 0, 30, 0)
        sat = st.slider("Job Satisfaction", 1, 5, 1)

    if st.button("Predict Attrition", type="primary", use_container_width=True):
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
        
        # Calculate engineered features
        new_data['projects_per_hour'] = new_data['projects_handled'] / new_data['avg_weekly_hours']
        new_data["absence_rate"] = new_data["absences_days"] / 365
        new_data["performance_efficiency"] = new_data["performance_rating"] / new_data["avg_weekly_hours"]
        new_data["stress_index"] = new_data["avg_weekly_hours"] * new_data["absences_days"]
        new_data["satisfaction_score"] = new_data["job_satisfaction"] * new_data["performance_rating"]
        
        new_data = new_data[feature_cols]
        scaled_data = scaler.transform(new_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        st.divider()
        st.subheader("Prediction Outcome")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prediction == 1:
                st.error("### ⚠️ YES\nEmployee is highly likely to leave.")
            else:
                st.success("### ✅ NO\nEmployee is likely to stay.")
            
            # Gauge chart for probability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                title = {'text': "Attrition Probability (%)"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': "red" if prediction == 1 else "green"}}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("**Interpretability: What is driving this prediction?**")
            # Calculate local feature importance: scaled input * model coefficients
            contributions = scaled_data[0] * model.coef_[0]
            contrib_df = pd.DataFrame({'Feature': feature_cols, 'Contribution': contributions})
            contrib_df = contrib_df.sort_values(by='Contribution', ascending=True)
            
            # Plotly horizontal bar chart
            fig_contrib = px.bar(
                contrib_df, 
                x='Contribution', 
                y='Feature', 
                orientation='h',
                color='Contribution',
                color_continuous_scale=px.colors.diverging.RdBu_r,
                title="Factors pushing toward Leaving (Red) vs. Staying (Blue)"
            )
            fig_contrib.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig_contrib, use_container_width=True)

# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab2:
    st.subheader("Explore the Training Dataset")
    st.write("Select a variable to see how it correlates with employee attrition in the historical data.")
    
    # Remove attrition from the selection options
    eda_features = [col for col in raw_df.columns if col not in ['employee_id', 'attrition']]
    selected_feature = st.selectbox("Select a Feature to Analyze:", eda_features)
    
    # Create an interactive Plotly histogram
    fig_eda = px.histogram(
        raw_df, 
        x=selected_feature, 
        color="attrition", 
        barmode="group",
        title=f"Distribution of {selected_feature.replace('_', ' ').title()} by Attrition",
        color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"}
    )
    st.plotly_chart(fig_eda, use_container_width=True)
    
    with st.expander("View Raw Data Preview"):
        st.dataframe(raw_df.head(50))

# ==========================================
# TAB 3: MODEL INSIGHTS
# ==========================================
with tab3:
    st.subheader("Global Feature Importance")
    st.write("This chart shows which features the Logistic Regression model relies on the most across all employees.")
    
    # Global coefficients
    global_coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_[0]})
    global_coef_df['Absolute_Importance'] = global_coef_df['Coefficient'].abs()
    global_coef_df = global_coef_df.sort_values(by='Absolute_Importance', ascending=True)
    
    fig_global = px.bar(
        global_coef_df, 
        x='Coefficient', 
        y='Feature', 
        orientation='h',
        color='Coefficient',
        color_continuous_scale=px.colors.diverging.RdBu_r,
    )
    st.plotly_chart(fig_global, use_container_width=True)
