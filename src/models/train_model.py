import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

from src.models.model import FraudDetectionModel

base_dir = Path(__file__).resolve().parents[2]
processed_dir = base_dir / "data" / "processed"
model_dir = base_dir / "src" / "models"
model_dir.mkdir(parents=True, exist_ok=True)

#Loading processed datasets
train_df = pd.read_csv(processed_dir / "train.csv")
test_df = pd.read_csv(processed_dir / "test.csv")

#Split features and labels
x_train = train_df.drop(columns=["Class"]).values
y_train = train_df["Class"].values

x_test = test_df.drop(columns=["Class"]).values
y_test = test_df["Class"].values

#Converting our data into PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

#Tests
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Initialize the model
input_dim = x_train.shape[1]
model = FraudDetectionModel(input_dim)

# Loss function for binary classification (with logits)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), model_dir / "fraud_detection_model.pt")