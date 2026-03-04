import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.dataset import CrisisDataset
from src.model import LSTMModel


def load_model(input_size):
    model = LSTMModel(input_size)
    model.load_state_dict(torch.load("models/crisis_model.pth"))
    model.eval()
    return model


def main():

    # Load dataset
    dataset = CrisisDataset("data/final/global_dataset.csv")

    X = dataset.X
    y = dataset.y

    input_size = X.shape[2]

    model = load_model(input_size)

    # Predict probabilities
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).numpy()

    # Load raw dataframe
    df = pd.read_csv("data/final/global_dataset.csv")

    # Align dates
    dates = pd.to_datetime(df.iloc[12:].iloc[:,0])

    # Load SP500
    sp500 = df.iloc[12:]["SP500"].values

    # Plot
    plt.figure(figsize=(14,7))

    plt.subplot(2,1,1)
    plt.plot(dates, sp500)
    plt.title("SP500 Market Trend")

    plt.subplot(2,1,2)
    plt.plot(dates, probs)
    plt.axhline(0.5, linestyle="--")
    plt.title("Predicted Crisis Probability")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()