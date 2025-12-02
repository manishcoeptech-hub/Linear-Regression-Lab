import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Linear Regression Demo: Predicting Marks from Study Hours")
st.write("This app demonstrates simple linear regression using a small dataset.")

# ---- DATASET ----
data = {
    "Hours": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "Marks": [35, 38, 45, 50, 55, 63, 70, 75]
}

df = pd.DataFrame(data)

st.subheader("Dataset")
st.dataframe(df)

# ---- MODEL TRAINING ----
X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

st.subheader("Regression Line")
fig, ax = plt.subplots()
ax.scatter(df["Hours"], df["Marks"], label="Actual Data")

# Best fit line
x_range = np.linspace(df["Hours"].min(), df["Hours"].max(), 100)
ax.plot(x_range, model.predict(x_range.reshape(-1, 1)), label="Regression Line")

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks Obtained")
ax.legend()
st.pyplot(fig)

# ---- USER INPUT ----
st.subheader("Predict Marks")
hours = st.slider("Select Study Hours", 0.0, 10.0, 2.0)

predicted_marks = model.predict([[hours]])[0]
st.write(f"**Predicted Marks:** {predicted_marks:.2f}")
