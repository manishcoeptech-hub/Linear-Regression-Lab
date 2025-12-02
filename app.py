import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.title("Linear Regression Lab (Enhanced Version)")
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
