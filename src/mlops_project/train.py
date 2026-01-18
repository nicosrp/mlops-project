import typer
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import datetime
import wandb
from pathlib import Path

from mlops_project.model import Model
from mlops_project.data import MyDataset

app = typer.Typer()

def train(
        dataset_path: str,
        epochs: int = 10,
        batch_size: int = 16,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3, 
        lr: float = 1e-3,
        seq_len: int = 10,
        model_path: str = None
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Weights and Biases logging
    wandb.init(
        project = "football-lstm",
        config={
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout
        }
    )

    try:
        dataset = MyDataset(Path(dataset_path), seq_len=seq_len)
        
        # Train/validation split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Model
        model = Model(
            input_size=dataset.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * X.size(0)
            
            avg_loss = total_loss / len(train_loader.dataset)
        
            # Validation
            model.eval()
            total_val_loss = 0
            correct, total = 0,0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)

                    output_val = model(X_val)

                    val_loss = criterion(output_val, y_val)
                    total_val_loss += val_loss.item() * X_val.size(0)

                    preds = output_val.argmax(dim=1)
                    correct += (preds == y_val).sum().item()
                    total += y_val.size(0)

                avg_val_loss = total_val_loss / len(val_loader.dataset)
                val_acc = correct / total

            print(f"Epoch {epoch}/{epochs} | Train loss: {avg_loss:.4f} | "
                  f"Val acc: {val_acc:.4f} | Val loss: {avg_val_loss: .4f}")
            
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc
            })

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    finally:
        wandb.finish()


# -------------------------
# CLI command
# -------------------------
@app.command()
def train_cli(
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="Path to preprocessed CSV"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size"),
    hidden_size: int = typer.Option(64, "--hidden-size", "-hs", help="Hidden size of LSTM"),
    num_layers: int = typer.Option(2, "--num-layers", "-nl", help="Number of LSTM layers"),
    dropout: float = typer.Option(0.3, "--dropout", "-do", help="Dropout rate"),
    lr: float = typer.Option(1e-3, "--lr", "-lr", help="Learning rate"),
    seq_len: int = typer.Option(10, "--seq-len", "-sl", help="Sequence length"),
    model_path: str = typer.Option(None, "--model-path", "-mp", help="Path to save model")
):
    """CLI entry point to train the LSTM model."""

    if model_path is None:
        # Default model path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/lstm_{timestamp}.pt"

    train(dataset_path, epochs, batch_size, hidden_size, num_layers, dropout, lr, seq_len, model_path)

if __name__ == "__main__":
    app()
