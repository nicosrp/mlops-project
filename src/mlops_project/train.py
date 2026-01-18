import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path
import logging

from mlops_project.model import Model
from mlops_project.data import MyDataset

log = logging.getLogger(__name__)

@hydra.main(
        config_path="../../configs",
        config_name="train.yaml"
)

def train(cfg: DictConfig) -> None:

    torch.manual_seed(cfg.training.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Weights and Biases logging
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    dataset = MyDataset(
        Path(cfg.paths.dataset),
        seq_len=cfg.hyperparameters.seq_len
    )
    
    # Train/validation split
    train_size = int(cfg.training.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.hyperparameters.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.hyperparameters.batch_size
    )

    # Model
    model = Model(
        input_size=dataset.input_size,
        hidden_size=cfg.hyperparameters.hidden_size,
        num_layers=cfg.hyperparameters.num_layers,
        dropout=cfg.hyperparameters.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.hyperparameters.lr
    )

    # Training loop
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        
        train_loss = total_loss / len(train_loader.dataset)
    
        # Validation
        model.eval()
        total_val_loss = 0
        correct, total = 0,0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                logits = model(x)

                val_loss = criterion(logits, y)
                total_val_loss += val_loss.item() * x.size(0)

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            val_acc = correct / total

        # Logging
        log.info(
            f"Epoch {epoch + 1}/{cfg.hyperparameters.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    # Save model
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "trained_model.pt"
    torch.save(model.state_dict(), model_path)

    log.info(f"Model saved to {model_path}")

    wandb.save(str(model_path))
    wandb.finish()


if __name__ == "__main__":
    train()
