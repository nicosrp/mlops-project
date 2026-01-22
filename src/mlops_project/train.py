import logging
from pathlib import Path

import hydra
import torch
import torch.nn.functional as func
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from mlops_project.data import MyDataset
from mlops_project.model import Model

log = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss to address class imbalance by focusing on hard examples.
    Helps prevent the model from just predicting the majority class.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = func.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


@hydra.main(config_path="../../configs", config_name="train.yaml", version_base=None)
def train(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Weights and Biases logging
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Update config with sweep parameters if they exist (for hyperparameter sweeps)
    if wandb.config:
        for key, value in dict(wandb.config).items():
            if key.startswith("hyperparameters."):
                param_name = key.split(".")[1]
                if param_name in cfg.hyperparameters:
                    cfg.hyperparameters[param_name] = value

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
        use_attention=cfg.hyperparameters.get("use_attention", True),
    ).to(device)

    log.info(f"Model input size: {dataset.input_size}")
    log.info(f"Using attention: {cfg.hyperparameters.get('use_attention', True)}")

    # Calculate class weights to handle imbalance
    # Class distribution: home=43.4%, away=31.7%, draw=24.9%
    import numpy as np

    unique, counts = np.unique(dataset.y, return_counts=True)
    total_samples = len(dataset.y)

    # Balanced weights with power 0.5 (sqrt) for smoother weighting
    class_weights = torch.tensor(
        [(total_samples / (len(unique) * count)) ** 0.5 for count in counts], dtype=torch.float32
    )
    class_weights = class_weights.to(device)
    log.info(f"Class weights: home={class_weights[0]:.3f}, draw={class_weights[1]:.3f}, away={class_weights[2]:.3f}")

    # Use Focal Loss for better handling of hard examples
    use_focal_loss = cfg.hyperparameters.get("use_focal_loss", True)
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        log.info("Using Focal Loss with gamma=2.0")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        log.info("Using CrossEntropy Loss")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

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

        # Track per-class accuracy
        class_correct = torch.zeros(3)
        class_total = torch.zeros(3)

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

                # Per-class accuracy
                for c in range(3):
                    class_mask = y == c
                    class_correct[c] += ((preds == y) & class_mask).sum().item()
                    class_total[c] += class_mask.sum().item()

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            val_acc = correct / total

            # Calculate per-class accuracies
            home_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
            draw_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0
            away_acc = class_correct[2] / class_total[2] if class_total[2] > 0 else 0

        # Logging
        log.info(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"Home acc: {home_acc:.4f} | Draw acc: {draw_acc:.4f} | Away acc: {away_acc:.4f}"
        )

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_acc_home": home_acc,
                "val_acc_draw": draw_acc,
                "val_acc_away": away_acc,
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
        "use_attention": cfg.hyperparameters.get("use_attention", True),
    }
    torch.save(checkpoint, model_path)

    log.info(f"Model saved to {model_path}")

    # Upload model to GCS if running on GCP
    import os
    import subprocess
    from datetime import datetime

    if os.getenv("CLOUD_ML_JOB_ID"):  # Running on GCP
        try:
            gcs_bucket = cfg.get("gcp", {}).get("bucket_models", "mlops-484822-models")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save to MULTIPLE locations for safety
            paths_to_upload = [
                (str(model_path), f"gs://{gcs_bucket}/best_model.pth"),
                (str(model_path), f"gs://{gcs_bucket}/backups/best_model_{timestamp}.pth"),
                (
                    str(model_path),
                    f"gs://{gcs_bucket}/trained_models/model_epoch{cfg.hyperparameters.epochs}_{timestamp}.pth",
                ),
            ]

            for local_path, gcs_path in paths_to_upload:
                log.info(f"Uploading model to {gcs_path}...")
                subprocess.run(["gsutil", "cp", local_path, gcs_path], check=True)
                log.info(f"✅ Model uploaded to GCS: {gcs_path}")

            log.info(f"🎉 Model saved to {len(paths_to_upload)} GCS locations successfully!")
        except Exception as e:
            log.error(f"❌ CRITICAL: Failed to upload model to GCS: {e}")
            raise  # Fail the job if we can't save the model

    wandb.save(str(model_path))
    wandb.finish()


if __name__ == "__main__":
    train()
