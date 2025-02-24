import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the app
st.title("CSV File Uploader and Chart Generator")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

def generate_bar(data, x_axis, y_axis):
    st.write("Bar Chart:")
    fig = px.bar(data, x=x_axis, y=y_axis)
    fig.update_layout(height=600, width=900)
    st.plotly_chart(fig)

def generate_line(data, x_axis, y_axis):
    st.write("Line Chart:")
    fig = px.line(data, x=x_axis, y=y_axis)
    fig.update_layout(height=600, width=900)
    st.plotly_chart(fig)

def generate_scatter(data, x_axis, y_axis):
    st.write("Scatter Plot:")
    fig = px.scatter(data, x=x_axis, y=y_axis)
    fig.update_layout(height=600, width=900)
    st.plotly_chart(fig)

def generate_pie(data, x_axis, y_axis):
    st.write("Pie Chart:")
    fig = px.pie(data, names=x_axis, values=y_axis)
    fig.update_layout(height=600, width=900)
    st.plotly_chart(fig)

def generate_histogram(data, x_axis):
    st.write("Histogram:")
    fig = px.histogram(data, x=x_axis)
    fig.update_layout(height=600, width=900)
    st.plotly_chart(fig)

def generate_box(data, x_axis, y_axis):
    st.write("Box Plot:")
    fig = px.box(data, x=x_axis, y=y_axis)
    fig.update_layout(height=600, width=900)
    st.plotly_chart(fig)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("File content:")
    st.write(df)

    # Select columns for plotting
    columns = df.columns.tolist()
    x_axis_col, y_axis_col = st.columns(2)
    with x_axis_col:
        x_axis = st.selectbox("Select X-axis column", columns)
    with y_axis_col:
        y_axis = st.selectbox("Select Y-axis column", columns)

    try:
        if x_axis == y_axis:
            raise ValueError("X-axis and Y-axis columns cannot be the same.")
    except ValueError as e:
        st.error(str(e))

    # Place buttons side by side
    button_col1, button_col2, button_col3, button_col4, button_col5, button_col6 = st.columns(6)

    with button_col1:
        bar_chart_button = st.button("Generate Bar Chart")
    with button_col2:
        line_chart_button = st.button("Generate Line Chart")
    with button_col3:
        scatter_plot_button = st.button("Generate Scatter Plot")
    with button_col4:
        pie_chart_button = st.button("Generate Pie Chart")
    with button_col5:
        histogram_button = st.button("Generate Histogram")
    with button_col6:
        box_plot_button = st.button("Generate Box Plot")

    # Generate charts based on clicked button
    if bar_chart_button:
        generate_bar(df, x_axis, y_axis)
    if line_chart_button:
        generate_line(df, x_axis, y_axis)
    if scatter_plot_button:
        generate_scatter(df, x_axis, y_axis)
    if pie_chart_button:
        generate_pie(df, x_axis, y_axis)
    if histogram_button:
        generate_histogram(df, x_axis)
    if box_plot_button:
        generate_box(df, x_axis, y_axis)
