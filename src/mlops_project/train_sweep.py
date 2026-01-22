import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from mlops_project.data import MyDataset
from mlops_project.model import Model

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="train.yaml", version_base=None)
def train(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Weights and Biases logging
    # For sweeps, wandb.init() is called by the sweep agent
    # We just need to merge the sweep config with our base config
    if wandb.run is None:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Update config with sweep parameters if they exist
    if wandb.config:
        for key, value in wandb.config.items():
            if "." in key:  # Handle nested keys like "hyperparameters.lr"
                parts = key.split(".")
                if parts[0] == "hyperparameters" and parts[1] in cfg.hyperparameters:
                    OmegaConf.update(cfg, key, value, merge=False)

    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Get absolute path to dataset
    data_path = Path(cfg.paths.dataset)
    if not data_path.is_absolute():
        # Hydra changes working directory, so use original path from hydra.utils
        from hydra.utils import get_original_cwd

        data_path = Path(get_original_cwd()) / data_path

    dataset = MyDataset(data_path, seq_len=cfg.hyperparameters.seq_len)

    # Train/validation split
    train_size = int(cfg.training.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.hyperparameters.batch_size)

    # Model
    model = Model(
        input_size=dataset.input_size,
        hidden_size=cfg.hyperparameters.hidden_size,
        num_layers=cfg.hyperparameters.num_layers,
        dropout=cfg.hyperparameters.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    # Training loop
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        total_loss = 0.0

        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.hyperparameters.epochs} [Train]", leave=False)
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg.hyperparameters.epochs} [Val]", leave=False)
            for x, y in val_pbar:
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
            f"Val loss: {avg_val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
            }
        )

    # Save model
    model_dir = Path(cfg.paths.model_dir)
    if not model_dir.is_absolute():
        from hydra.utils import get_original_cwd

        model_dir = Path(get_original_cwd()) / model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best_model.pth"

    # Save full checkpoint with hyperparameters
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": dataset.input_size,
        "hidden_size": cfg.hyperparameters.hidden_size,
        "num_layers": cfg.hyperparameters.num_layers,
        "dropout": cfg.hyperparameters.dropout,
    }
    torch.save(checkpoint, model_path)

    log.info(f"Model saved to {model_path}")

    wandb.save(str(model_path))
    wandb.finish()


if __name__ == "__main__":
    train()
