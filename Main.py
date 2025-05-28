import streamlit as st

st.title("💼 Loan Repayment Prediction & Financial Analysis")

st.markdown("""
    Welcome to the Loan Analytics Dashboard! This tool uses real loan data to:

    - 📊 Predict loan defaults using logistic regression
    - 📈 Visualize key financial metrics like disbursement and repayment
    - 🧠 Assess client behavior and financial literacy risk
    - 📁 Explore repayment patterns and business loan performance

    ### 🔄 Get Started:
    1. Upload a `.csv` file containing your loan data
    2. View automatic charts and statistics
    3. Check the **About Us** section for more information

    _Need help with formatting your data? Make sure your CSV includes columns like `lnDisbursed`, `lnBalance`, `loan_pledge_amt`, `term`, and `Default`._
    """)



