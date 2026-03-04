import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.dataset import CrisisDataset
from src.model import LSTMModel


def main():

    print("TRAINING STARTED")

    # Load dataset
    dataset = CrisisDataset("data/final/global_dataset.csv")

    # Train/Test split
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    input_size = dataset.X.shape[2]

    # Model
    model = LSTMModel(input_size)

    # Handle class imbalance
    labels = dataset.y.numpy()
    pos = labels.sum()
    neg = len(labels) - pos

    pos_weight = torch.tensor([neg / pos])

    print("Positive samples:", pos)
    print("Negative samples:", neg)
    print("Class weight (pos_weight):", pos_weight.item())

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    # ==========================
    # Training Loop
    # ==========================

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            outputs = model(X_batch).squeeze()

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    print("\nEvaluating on Test Set...")

    # ==========================
    # Evaluation
    # ==========================

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            outputs = model(X_batch).squeeze()

            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).float()

            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())
            y_prob.extend(probs.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== TEST RESULTS =====")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    # ==========================
    # Save Model
    # ==========================

    import os

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/crisis_model.pth")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()