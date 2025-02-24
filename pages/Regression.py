import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def preprocess_data(df):
    # Remove commas and convert to numeric
    numeric_cols = ['lnDisbursed', 'lnBalance', 'loan_pledge_amt']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    # Compute percentage
    df['Percentage'] = (df['lnDisbursed'] - df['lnBalance']) / df['lnDisbursed']

    # Select relevant columns
    df = df[['clientName', 'lnDisbursed', 'lnBalance', 'disbDate', 'maturity', 'term', 'loan_pledge_amt', 'Percentage', 'acctclass', 'business', 'Default']].copy()

    # Drop missing values
    df.dropna(inplace=True)

    return df


def train_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Ensure 'Default' is numeric and binary
    df['Default'] = pd.to_numeric(df['Default'], errors='coerce')

    # Convert 'Default' to binary (1 if there’s any remaining balance, 0 otherwise)
    df['Default'] = df['Default'].apply(lambda x: 1 if x > 0 else 0)

    # Drop NaN values in 'Default'
    df = df.dropna(subset=['Default'])

    # Debugging: Print unique values
    unique_values = df['Default'].unique()
    print(f"Unique values in 'Default' after conversion: {unique_values}")

    if len(unique_values) > 2:
        print("Error: 'Default' column still has more than two unique values! Fix your dataset.")
        return None, None, None

    # Features (X) and Target (y)
    X = df[['lnDisbursed', 'lnBalance', 'term', 'loan_pledge_amt', 'Percentage']]
    y = df['Default']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, accuracy


def financial_literacy_assessment(df):
    df['Risk Score'] = df['Percentage'].apply(lambda x: 'High Risk' if x < 0.3 else ('Medium Risk' if x < 0.7 else 'Low Risk'))
    return df[['clientName', 'Risk Score']]

def insurance_recommendation(df):
    df['Insurance Plan'] = df['Percentage'].apply(lambda x: 'High Coverage' if x < 0.3 else ('Medium Coverage' if x < 0.7 else 'Basic Coverage'))
    return df[['clientName', 'Insurance Plan']]


st.title('Loan Repayment Prediction & Financial Analysis')
uploaded_file = st.file_uploader('Upload CSV File', type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    df = preprocess_data(df)
    st.write('### Sample Data')
    st.dataframe(df.head())
    
    # Data Visualization
    st.write('### Loan Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['lnDisbursed'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.write('### Loan Amount vs Remaining Balance')
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['lnDisbursed'], y=df['lnBalance'], hue=df['Default'], ax=ax)
    st.pyplot(fig)
    
    # Train Model
    model, scaler, accuracy = train_model(df)
    if accuracy is not None:
        st.write(f'### Model Accuracy: {accuracy * 100:.2f}%')
    else:
        st.error("Model training failed. Check the dataset for invalid values in 'Default'.")

    # Histogram
    st.write("### Loan Repayment Percentage Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Percentage'], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Repayment Percentage")
    ax.set_title("Histogram of Loan Repayment Percentage")
    st.pyplot(fig)

    # Financial Literacy & Insurance
    st.write('### Financial Literacy Assessment')
    st.dataframe(financial_literacy_assessment(df))
    
    st.write('### Insurance Recommendations')
    st.dataframe(insurance_recommendation(df))
        
