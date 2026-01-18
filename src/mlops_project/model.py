from torch import nn
import torch

class Model(nn.Module):
    """
    LSTM-based model for predicting match outcome using sequence data.
    Input shape: (batch_size, seq_len, feature_dim)
    seq_len is the number of historical matches (ranging from 0-10).
    feature_dim is the number of features per match.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            output_size: int = 3,    # Number of classes: home/draw/away
            dropout: float = 0.3
    ):
        super().__init__()
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # FC layer for output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        # LSTM output: (batch_size, seq_len, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]   # shape (batch_size, hidden_size)

        return self.fc(last_hidden)

if __name__ == "__main__":
    # Dummy block
    batch_size = 1
    seq_len = 10
    feature_dim = 18

    model = Model(input_size=feature_dim)
    x = torch.rand(batch_size, seq_len, feature_dim)
    out = model(x)

    print(f"Output shape of model: {out.shape}")
