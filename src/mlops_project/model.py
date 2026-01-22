import torch
from torch import nn
import math


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
            dropout: float = 0.3,
            use_attention: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism to focus on important matches
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

        # Enhanced FC layer with batch normalization
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # LSTM output: (batch_size, seq_len, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        
        if self.use_attention:
            # Apply attention mechanism to focus on important matches
            # out shape: (batch_size, seq_len, hidden_size)
            attention_scores = self.attention(out)  # (batch_size, seq_len, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)
            
            # Weighted sum of LSTM outputs
            context = torch.sum(attention_weights * out, dim=1)  # (batch_size, hidden_size)
        else:
            # Use last hidden state
            context = h_n[-1]  # (batch_size, hidden_size)

        return self.fc(context)

if __name__ == "__main__":
    # Dummy block
    batch_size = 1
    seq_len = 10
    feature_dim = 18

    model = Model(input_size=feature_dim)
    x = torch.rand(batch_size, seq_len, feature_dim)
    out = model(x)

    print(f"Input shape of model: {x.shape}")
    print(f"Output shape of model: {out.shape}")
