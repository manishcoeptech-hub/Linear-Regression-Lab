import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.title("Linear Regression Lab")

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
theory_tab, lab_tab = st.tabs(["📘 Theory", "🧪 Lab Activity"])

# -------------------------------------------------------------------------
# THEORY TAB CONTENT
# -------------------------------------------------------------------------
with theory_tab:
    st.header("📘 Theory of Linear Regression")

    st.markdown("""
###  What is Linear Regression?

Linear Regression is a statistical method used to model the relationship between:

- **Independent variable (X)** → Example: Hours studied  
- **Dependent variable (Y)** → Example: Marks obtained  

The goal is to draw a **straight line** that best fits the data.
""")

    st.markdown("---")

    st.markdown("###  Equation of the Line")
    st.latex(r"y = mx + c")

    st.markdown("""
### **Where:**
- $m$ = slope of the line  
- $c$ = intercept  
- $y$ = predicted output  
""")

    st.markdown("---")
    st.header("📊 Error Metrics (Model Accuracy Measures)")

    # ---------------- MAE ----------------
    st.subheader("1️. MAE — Mean Absolute Error")

    st.markdown("### Formula")
    st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")

    st.markdown("""
### **Where:**
- $y_i$ = actual value  
- $\hat{y}_i$ = predicted value  
- $n$ = total number of data points  
- $|y_i - \hat{y}_i|$ = absolute error  
""")

    st.markdown("""
### **Explanation**
- Measures the **average absolute difference** between actual and predicted values  
- Easy to understand  
- Lower MAE = better accuracy  
""")

    st.markdown("---")

    # ---------------- MSE ----------------
    st.subheader("2️. MSE — Mean Squared Error")

    st.markdown("### Formula")
    st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

    st.markdown("""
### **Where:**
- $y_i$ = actual value  
- $\hat{y}_i$ = predicted value  
- $n$ = total number of samples  
- $(y_i - \hat{y}_i)^2$ = squared error  
""")

    st.markdown("""
### **Explanation**
- Squares the errors → **penalizes large mistakes more**  
- Always ≥ 0  
- Lower MSE = better model  
""")

    st.markdown("---")

    # ---------------- R2 ----------------
    st.subheader("3️. R² Score — Coefficient of Determination")

    st.markdown("### Formula")
    st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")

    st.markdown("""
### **Where:**
- $y_i$ = actual value  
- $\hat{y}_i$ = predicted value  
- $\\bar{y}$ = mean of actual values  
- Numerator = squared prediction errors  
- Denominator = squared deviations from mean  
""")

    st.markdown("""
### **Explanation**
- Measures how well the regression line fits the data  
- R² = 1 → perfect fit  
- R² = 0 → no better than average prediction  
- R² < 0 → very poor model  
""")

    st.markdown("---")

    st.markdown("""
### Real-Life Uses
- Predicting student marks  
- Forecasting sales  
- Estimating house prices  
- Weather prediction  
- Business trend analysis  
""")

# -------------------------------------------------------------------------
# LAB TAB
# -------------------------------------------------------------------------
with lab_tab:
    
    st.write("Upload a CSV file containing two columns: Hours and Marks.")

    # ---- File Upload ----
    uploaded_file = st.file_uploader("Upload dataset (.csv)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("Your Dataset")
        st.dataframe(df)

        # ---- Train Model ----
        X = df[["Hours"]]
        y = df["Marks"]

        model = LinearRegression()
        model.fit(X, y)

        m = model.coef_[0]       # slope
        c = model.intercept_     # intercept

        # ---- Predictions for metric calculation ----
        y_pred = model.predict(X)

        # ---- Error Metrics ----
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.subheader(" Model Accuracy Metrics")
        st.write(f"**MAE (Mean Absolute Error):** {mae:.3f}")
        st.write(f"**MSE (Mean Squared Error):** {mse:.3f}")
        st.write(f"**R² Score:** {r2:.3f}")

        # ---- Show Actual Equation ----
        st.subheader("📌 Regression Equation (Using Your Dataset)")
        st.latex(fr"y = {m:.3f}x + {c:.3f}")

        # ---- Show Calculations Button ----
        if st.button(" Show Calculations for m and c"):
            st.markdown("### 🔍 Step-by-Step Calculation")

            # Manual calculations
            x_mean = df["Hours"].mean()
            y_mean = df["Marks"].mean()

            df["(x - x̄)"] = df["Hours"] - x_mean
            df["(y - ȳ)"] = df["Marks"] - y_mean
            df["(x - x̄)(y - ȳ)"] = df["(x - x̄)"] * df["(y - ȳ)"]
            df["(x - x̄)²"] = df["(x - x̄)"] ** 2

            st.markdown("### Table Used for Calculations")
            st.dataframe(df)

            numerator = df["(x - x̄)(y - ȳ)"].sum()
            denominator = df["(x - x̄)²"].sum()

            st.markdown("### Formulas")
            st.latex(r"m = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}")
            st.latex(r"c = \bar{y} - m\bar{x}")

            st.markdown("### **Values Used**")
            st.write(f"**Mean of Hours (x̄):** {x_mean:.3f}")
            st.write(f"**Mean of Marks (ȳ):** {y_mean:.3f}")
            st.write(f"**Σ(x - x̄)(y - ȳ):** {numerator:.3f}")
            st.write(f"**Σ(x - x̄)²:** {denominator:.3f}")

            m_calc = numerator / denominator
            c_calc = y_mean - m_calc * x_mean

            st.markdown("### 🎯 Final Calculated Values")
            st.write(f"**Slope (m):** {m_calc:.3f}")
            st.write(f"**Intercept (c):** {c_calc:.3f}")

        # ---- Regression Line Plot ----
        st.subheader("📈 Regression Line & Data Points")

        fig, ax = plt.subplots()
        ax.scatter(df["Hours"], df["Marks"], label="Actual Data", color="blue")

        x_range = np.linspace(df["Hours"].min(), df["Hours"].max(), 100)
        ax.plot(x_range, model.predict(x_range.reshape(-1, 1)),
                label="Regression Line", color="green")

        # ---- User Input for Prediction ----
        st.subheader("🔮 Predict Marks")

        user_hours = st.number_input(
            "Enter study hours:", min_value=0.0, max_value=24.0, step=0.5
        )

        if user_hours > 0:
            predicted_marks = model.predict([[user_hours]])[0]
            st.success(f"Predicted Marks: **{predicted_marks:.2f}**")

            # Add the predicted point to the plot
            ax.scatter(user_hours, predicted_marks, color="red",
                       s=100, label="Predicted Point")

            ax.legend()
            st.pyplot(fig)

        else:
            ax.legend()
            st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to continue.")
