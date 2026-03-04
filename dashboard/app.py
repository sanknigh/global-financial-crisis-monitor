import sys
import os

# Add project root so Streamlit can see src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import torch

from src.dataset import CrisisDataset
from src.model import LSTMModel


st.title("🌍 Global Financial Crisis Monitor")


# Load dataset
dataset = CrisisDataset("data/final/global_dataset.csv")

X = dataset.X
y = dataset.y


# Load trained model
input_size = X.shape[2]
model = LSTMModel(input_size)

model.load_state_dict(torch.load("models/crisis_model.pth"))
model.eval()


# Predict probabilities
with torch.no_grad():
    probs = torch.sigmoid(model(X)).numpy().flatten()


# Load dataframe
df = pd.read_csv("data/final/global_dataset.csv")

dates = pd.to_datetime(df.iloc[12:, 0])


# Create dataframe for plotting
plot_df = pd.DataFrame({
    "Date": dates,
    "Crisis_Probability": probs
})


# Plot crisis probability
fig = px.line(
    plot_df,
    x="Date",
    y="Crisis_Probability",
    title="Predicted Financial Crisis Probability"
)

st.plotly_chart(fig, use_container_width=True)


# Risk indicator
latest_prob = probs[-1]

st.subheader("Current Crisis Risk")

if latest_prob > 0.7:
    st.error(f"High Risk ⚠️ ({latest_prob:.2f})")
elif latest_prob > 0.4:
    st.warning(f"Medium Risk ({latest_prob:.2f})")
else:
    st.success(f"Low Risk ({latest_prob:.2f})")