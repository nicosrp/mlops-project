import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import wandb
from pathlib import Path

from mlops_project.model import Model
from mlops_project.data import MyDataset

# ======================
# Training configuration
# ======================
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 10
SEQ_LEN = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Weights and Biases logging
    wandb.init(project = "football-lstm", config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT
    })

    dataset_path = Path("data/processed/processed_data.csv")
    dataset = MyDataset(dataset_path, seq_len=10)
    
    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model
    model = Model(
        input_size=dataset.input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
    
        # Validation
        model.eval()
        correct, total = 0,0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                output_val = model(X_val)
                preds = output_val.argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
            val_acc = correct / total

        print(f"Epoch {epoch}/{EPOCHS} | Train loss: {avg_loss:.4f} | Val acc: {val_acc:.4f}")
        wandb.log({"train_loss": avg_loss, "val_acc": val_acc})

    model_path = Path("models/lstm.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    wandb.finish()

if __name__ == "__main__":
    train()
