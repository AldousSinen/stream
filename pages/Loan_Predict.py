import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Streamlit UI
st.title("📊 Loan Default Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file, encoding='utf-8')

    # Select necessary columns
    columns_needed = ['Loan', 'Balance', 'Paid', 'Savings', 'Interest', 'TermInWeeks', 'Age', 'LoanStatus']
    df = df[columns_needed]
    print("DF", df)
    # Convert LoanStatus: Create Default (1) & Paid (0) Labels
    def categorize_loan_status(row):
        if row['Paid'] / row['Loan'] < 0.5 and row['Balance'] > 0 and row['TermInWeeks'] < 0:
            return 1  # Default
        elif row['Paid'] / row['Loan'] >= 0.9:
            return 0  # Fully Paid
        else:
            return -1  # Active Loan (ignore)

    df['LoanStatus'] = df.apply(categorize_loan_status, axis=1)

    # Remove Active Loans (-1) since we can't use them for training
    df = df[df['LoanStatus'] != -1]

    # Feature Engineering: Create new risk-related features
    df['Interest_to_Loan_Ratio'] = df['Interest'] / df['Loan']
    df['Payment_Ratio'] = df['Paid'] / df['Loan']

    # Handle missing values
    df.fillna(df.median(), inplace=True)

    # Define feature set
    X = df.drop(columns=['LoanStatus'])
    y = df['LoanStatus']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy
    st.write(f"**Model Accuracy:** {accuracy:.2f}")

    # Show classification report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Visualization - Loan Balance vs Default
    st.subheader("Loan Balance Distribution")
    fig, ax = plt.subplots()
    df[df['LoanStatus'] == 1]['Balance'].hist(bins=20, color='red', alpha=0.7, label='Defaulted')
    df[df['LoanStatus'] == 0]['Balance'].hist(bins=20, color='green', alpha=0.7, label='Paid')
    ax.set_xlabel("Loan Balance")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

    # Prediction Form
    st.subheader("📌 Predict Loan Default for a New Customer")

    loan = st.number_input("Loan Amount", value=5000.0)
    balance = st.number_input("Current Balance", value=1000.0)
    paid = st.number_input("Total Paid", value=500.0)
    savings = st.number_input("Savings", value=200.0)
    interest = st.number_input("Interest Amount ($)", value=500.0)  # Updated to interest amount
    term = st.number_input("Loan Term (Weeks)", value=12)
    age = st.number_input("Age", value=30)

    if st.button("Predict Default Risk"):
        # Compute additional features
        interest_to_loan_ratio = interest / loan if loan > 0 else 0
        payment_ratio = paid / loan if loan > 0 else 0

        # Create input dataframe
        input_data = pd.DataFrame([[loan, balance, paid, savings, interest, term, age, interest_to_loan_ratio, payment_ratio]],
                                  columns=['Loan', 'Balance', 'Paid', 'Savings', 'Interest', 'TermInWeeks', 'Age',
                                           'Interest_to_Loan_Ratio', 'Payment_Ratio'])

        prediction = model.predict(input_data)[0]
        result = "🔴 High Default Risk" if prediction == 1 else "🟢 Low Default Risk"
        st.subheader(f"Prediction: {result}")


