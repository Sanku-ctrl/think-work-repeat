import numpy as np
import torch
from torch import nn

class StudyTimeModel(nn.Module):
    def __init__(self, input_size: int = 7) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 220,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> StudyTimeModel:
    torch.manual_seed(seed)

    model = StudyTimeModel(input_size=x_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        predictions = model(x_tensor)
        loss = criterion(predictions, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

@torch.no_grad()
def predict_minutes(model: StudyTimeModel, input_features: np.ndarray) -> int:
    model.eval()
    input_tensor = torch.tensor(input_features, dtype=torch.float32).reshape(1, -1)
    output = float(model(input_tensor).item())
    return max(1, int(round(output)))