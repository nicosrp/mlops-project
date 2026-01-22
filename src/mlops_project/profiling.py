"""Profiling script for training performance analysis."""

import cProfile
import pstats
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mlops_project.data import MyDataset
from mlops_project.model import Model


def profile_training():
    """Profile a single training iteration to identify bottlenecks."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MyDataset(Path("data/processed/processed_data.csv"), seq_len=10)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = Model(input_size=dataset.input_size, hidden_size=64, num_layers=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Profile one epoch
    def train_one_epoch():
        model.train()
        for i, (x, y) in enumerate(dataloader):
            if i >= 10:  # Only profile first 10 batches
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    train_one_epoch()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    profile_training()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")

    print("\n" + "=" * 80)
    print("PROFILING RESULTS - Top 20 functions by cumulative time")
    print("=" * 80)
    stats.print_stats(20)

    # Save to file
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    stats.dump_stats(output_dir / "profile_stats.prof")
    print(f"\nProfile stats saved to {output_dir / 'profile_stats.prof'}")
