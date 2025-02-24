import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Streamlit App Title
st.title("Loan Repayment Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())
    
    # Select relevant columns
    relevant_columns = ["Loan", "Balance", "Paid", "Savings", "Interest", "LoanStatus"]
    df = df[relevant_columns].dropna()
    
    # Encode LoanStatus (Assuming 'Paid' = 1 and 'Unpaid' = 0)
    df['LoanStatus'] = df['LoanStatus'].apply(lambda x: 1 if str(x).lower() == 'paid' else 0)
    
    # Data Visualization
    st.write("### Data Visualization")
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Loan', y='Balance', hue='LoanStatus', ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='LoanStatus', y='Paid', ax=ax)
    st.pyplot(fig)
    
    # Model Training
    st.write("### Train Model")
    X = df.drop(columns=['LoanStatus'])
    y = df['LoanStatus']
    
    # Standardize data for models that require scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(probability=True)
    }
    
    selected_model_name = st.selectbox("Choose a Model", list(models.keys()))
    selected_model = models[selected_model_name]
    
    selected_model.fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy ({selected_model_name}): {accuracy:.2f}")
    
    # Prediction for New Clients
    st.write("### Predict Loan Repayment for New Clients")
    loan = st.number_input("Loan Amount", min_value=0.0)
    balance = st.number_input("Balance", min_value=0.0)
    paid = st.number_input("Paid Amount", min_value=0.0)
    savings = st.number_input("Savings Amount", min_value=0.0)
    interest = st.number_input("Interest Amount", min_value=0.0)
    
    if st.button("Predict"):
        new_data = pd.DataFrame([[loan, balance, paid, savings, interest]], columns=X.columns)
        new_data_scaled = scaler.transform(new_data)
        prediction = selected_model.predict(new_data_scaled)[0]
        result = "Will Repay Loan" if prediction == 1 else "May Not Repay Loan"
        st.write(f"Prediction: {result}")
