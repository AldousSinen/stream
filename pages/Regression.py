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
    numeric_cols = ['lnDisbursed', 'lnBalance', 'loan_pledge_amt']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    df['Percentage'] = (df['lnDisbursed'] - df['lnBalance']) / df['lnDisbursed']
    df.dropna(inplace=True)
    return df

def train_model(df):
    df['Default'] = pd.to_numeric(df['Default'], errors='coerce')
    df['Default'] = df['Default'].apply(lambda x: 1 if x > 0 else 0)
    df = df.dropna(subset=['Default'])
    
    X = df[['lnDisbursed', 'lnBalance', 'term', 'loan_pledge_amt', 'Percentage']]
    y = df['Default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy

def financial_literacy_assessment(df):
    grouped_df = df.groupby('clientName').agg(
        avg_percentage=('Percentage', 'mean'),
        total_loans=('clientName', 'count'),
        business_loans=('prodType', lambda x: (x == 'MF Sikap 1').sum())
    ).reset_index()
    
    def classify_risk(percentage):
        if percentage < 0.3:
            return 'High Risk'
        elif percentage < 0.7:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    grouped_df['Risk Score'] = grouped_df['avg_percentage'].apply(classify_risk)
    return grouped_df[['clientName', 'avg_percentage', 'total_loans', 'business_loans', 'Risk Score']]

st.title('Loan Repayment Prediction & Financial Analysis')
uploaded_file = st.file_uploader('Upload CSV File', type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    df = preprocess_data(df)
    st.write('### Sample Data')
    st.dataframe(df)
    
    st.write('### Loan Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['lnDisbursed'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.write('### Loan Amount vs Remaining Balance')
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['lnDisbursed'], y=df['lnBalance'], hue=df['Default'], ax=ax)
    st.pyplot(fig)
    
    st.write("### Loan Repayment Percentage Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Percentage'], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Repayment Percentage")
    ax.set_title("Histogram of Loan Repayment Percentage")
    st.pyplot(fig)
    
    st.write("### Probability Distribution of Loan Amount")
    fig, ax = plt.subplots()
    sns.kdeplot(df['lnDisbursed'], fill=True, ax=ax)
    ax.set_xlabel("Loan Disbursed")
    ax.set_title("Probability Distribution of Loan Amount")
    st.pyplot(fig)
    
    model, scaler, accuracy = train_model(df)
    if accuracy is not None:
        st.write(f'### Model Accuracy: {accuracy * 100:.2f}%')
    else:
        st.error("Model training failed. Check dataset.")
    
    st.write('### Financial Literacy Assessment (Grouped by Client)')
    st.dataframe(financial_literacy_assessment(df))
