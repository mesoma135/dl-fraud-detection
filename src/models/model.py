import torch
import torch.nn as nn

class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            # Expanding all input features to learn initial fraud patterns
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Compress learned features into a more abstract representation
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # we should output a single logit for binary classification
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)