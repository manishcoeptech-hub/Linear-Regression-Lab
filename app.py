import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Linear Regression Lab (No Coding Required)")
st.write("Upload a CSV with two columns: Hours, Marks")

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Your Dataset")
    st.dataframe(df)

    # ---- Scatter Plot ----
    st.subheader("Scatter Plot")
    fig, ax = plt.subplots()
    ax.scatter(df["Hours"], df["Marks"])
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Marks Obtained")
    st.pyplot(fig)

    # ---- Train Model ----
    X = df[["Hours"]]
    y = df["Marks"]

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    st.write(f"**Equation of Line:** Marks = {slope:.2f} × Hours + {intercept:.2f}")

    # ---- Regression Line ----
    st.subheader("Regression Line")
    fig, ax = plt.subplots()
    ax.scatter(df["Hours"], df["Marks"], label="Data")
    x_range = np.linspace(df["Hours"].min(), df["Hours"].max(), 100)
    ax.plot(x_range, model.predict(x_range.reshape(-1, 1)), label="Regression Line")
    ax.legend()
    st.pyplot(fig)

    # ---- Prediction ----
    st.subheader("Predict Marks")
    hours = st.slider("Choose hours studied", 0.0, 10.0, 2.0)
    pred = model.predict([[hours]])[0]
    st.write(f"Predicted Marks: **{pred:.2f}**")

    # ---- Reflection Questions ----
    st.subheader("Answer These Questions in Your Lab Record:")
    st.markdown("""
    1. What is the slope of your regression line? What does it tell you?
    2. If you double the study hours, do the marks double? Why or why not?
    3. Add an outlier (e.g., Hours=1, Marks=95). How does the line change?
    4. Does your dataset show a strong or weak correlation? Explain.
    5. What are the limitations of predicting marks with linear regression?
    """)
else:
    st.info("Please upload a CSV file to continue.")
