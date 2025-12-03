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

Linear Regression is a statistical method that models the relationship between:

- **Independent variable (X)** → Example: Hours studied  
- **Dependent variable (Y)** → Example: Marks obtained  

The goal is to draw a **straight line** that best fits the data.

---

###  Equation of the Line

\[
y = mx + c
\]

Where:  
- **m** = slope  
- **c** = intercept  

---

###  Why Use Linear Regression?

- To **predict** future outcomes  
- To find **relationships** between variables  
- To analyze **trends** in data  

---

###  Error Metrics

#### **1. MAE (Mean Absolute Error)**  
Measures average absolute errors. Lower = better.

#### **2. MSE (Mean Squared Error)**  
Squares the errors, penalizes larger mistakes more.

#### **3. R² Score**  
Represents how well the model fits the data.  
- R² = 1 → Perfect  
- R² = 0 → No relationship  
- R² < 0 → Very poor model

---

###  Real-Life Uses
- Predicting marks  
- Forecasting sales  
- Estimating house prices  
- Weather prediction  
- Business forecasting  

---
    """)

# -------------------------------------------------------------------------
# LAB TAB (YOUR ORIGINAL CODE)
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

        # ---- Predictions for metric calculation ----
        y_pred = model.predict(X)

        # ---- Error Metrics ----
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.subheader("📊 Model Accuracy Metrics")
        st.write(f"**MAE (Mean Absolute Error):** {mae:.3f}")
        st.write(f"**MSE (Mean Squared Error):** {mse:.3f}")
        st.write(f"**R² Score:** {r2:.3f}")

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

            # Display updated plot
            st.pyplot(fig)
        else:
            ax.legend()
            st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to continue.")
