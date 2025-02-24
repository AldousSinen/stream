import streamlit as st
import pandas as pd

st.title("About Us")
data = pd.read_csv("tips.csv")
st.table(data)

st.bar_chart(data["total_bill"],)