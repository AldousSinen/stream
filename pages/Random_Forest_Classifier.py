import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit UI
st.title("📊 Loan Repayment Prediction App (Regression)")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file, encoding='utf-8')

    # Select necessary columns
    columns_needed = ['lnDisbursed', 'lnBalance', 'loan_pledge_amt', 'term', 'Percentage']
    df = df[columns_needed]

    # ✅ Remove percentage symbols and convert to float
    df['Percentage'] = df['Percentage'].astype(str).str.replace('%', '').astype(float) / 100

    # ✅ Ensure all columns are numeric
    for col in ['lnDisbursed', 'lnBalance', 'loan_pledge_amt', 'term']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing or invalid values
    df.dropna(inplace=True)

    # Feature Engineering: Additional risk-related features
    df['Savings_to_Loan_Ratio'] = df['loan_pledge_amt'] / df['lnDisbursed']
    df['Pledge_Impact'] = np.where(df['loan_pledge_amt'] >= df['lnBalance'], 1, 0)

    # Define feature set and target variable
    X = df.drop(columns=['Percentage'])
    y = df['Percentage']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display model performance
    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # Visualization - Actual vs Predicted
    st.subheader("Actual vs Predicted Repayment Percentage")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.plot([0, 1], [0, 1], '--', color='red')  # Perfect prediction line
    ax.set_xlabel("Actual Percentage")
    ax.set_ylabel("Predicted Percentage")
    ax.set_title("Regression Performance")
    st.pyplot(fig)

    # Prediction Form
    st.subheader("📌 Predict Loan Repayment Percentage for a New Customer")

    loan = st.number_input("Loan Amount", value=5000.0)
    balance = st.number_input("Current Balance", value=1000.0)
    savings = st.number_input("Pledged Savings", value=500.0)
    term = st.number_input("Loan Term (Weeks)", value=12)
    savings_to_loan_ratio = savings / loan if loan > 0 else 0
    pledge_impact = 1 if savings >= balance else 0

    if st.button("Predict Repayment Percentage"):
        # Create input dataframe
        input_data = pd.DataFrame([[loan, balance, savings, term, savings_to_loan_ratio, pledge_impact]],
                                  columns=['lnDisbursed', 'lnBalance', 'loan_pledge_amt', 'term', 'Savings_to_Loan_Ratio', 'Pledge_Impact'])

        prediction = model.predict(input_data)[0] * 100  # Convert back to percentage
        st.subheader(f"Predicted Repayment Percentage: {prediction:.2f}%")
