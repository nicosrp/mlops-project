# Activate venv and run all 5 sweep configurations
$ErrorActionPreference = "Stop"

Write-Host "================================================================================`n" -ForegroundColor Cyan
Write-Host "HYPERPARAMETER SWEEP - Running 5 configurations`n" -ForegroundColor Cyan
Write-Host "Each run: 10 epochs on full dataset`n" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

# Activate venv
& ".\venv\Scripts\Activate.ps1"

# Verify Python
$pythonPath = python -c "import sys; print(sys.executable)"
Write-Host "Using Python: $pythonPath`n"

# Config 1: Small model
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "RUN 1/5 - Small Model" -ForegroundColor Yellow
Write-Host "================================================================================`n" -ForegroundColor Yellow
python src/mlops_project/train.py hyperparameters.batch_size=32 hyperparameters.lr=0.0005 hyperparameters.hidden_size=32 hyperparameters.num_layers=1 hyperparameters.dropout=0.2 hyperparameters.seq_len=10 hyperparameters.epochs=10

# Config 2: Medium model
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "RUN 2/5 - Medium Model" -ForegroundColor Yellow
Write-Host "================================================================================`n" -ForegroundColor Yellow
python src/mlops_project/train.py hyperparameters.batch_size=16 hyperparameters.lr=0.001 hyperparameters.hidden_size=64 hyperparameters.num_layers=2 hyperparameters.dropout=0.3 hyperparameters.seq_len=10 hyperparameters.epochs=10

# Config 3: Large model
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "RUN 3/5 - Large Model" -ForegroundColor Yellow
Write-Host "================================================================================`n" -ForegroundColor Yellow
python src/mlops_project/train.py hyperparameters.batch_size=32 hyperparameters.lr=0.002 hyperparameters.hidden_size=128 hyperparameters.num_layers=2 hyperparameters.dropout=0.3 hyperparameters.seq_len=10 hyperparameters.epochs=10

# Config 4: Deep model
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "RUN 4/5 - Deep Model" -ForegroundColor Yellow
Write-Host "================================================================================`n" -ForegroundColor Yellow
python src/mlops_project/train.py hyperparameters.batch_size=16 hyperparameters.lr=0.0005 hyperparameters.hidden_size=64 hyperparameters.num_layers=3 hyperparameters.dropout=0.4 hyperparameters.seq_len=10 hyperparameters.epochs=10

# Config 5: High dropout model (test regularization)
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "RUN 5/5 - High Dropout Model" -ForegroundColor Yellow
Write-Host "================================================================================`n" -ForegroundColor Yellow
python src/mlops_project/train.py hyperparameters.batch_size=16 hyperparameters.lr=0.001 hyperparameters.hidden_size=128 hyperparameters.num_layers=2 hyperparameters.dropout=0.5 hyperparameters.seq_len=10 hyperparameters.epochs=10

Write-Host "`n================================================================================`n" -ForegroundColor Green
Write-Host "ALL SWEEP RUNS COMPLETED!`n" -ForegroundColor Green
Write-Host "================================================================================`n" -ForegroundColor Green
Write-Host "View results at:`n"
Write-Host "https://wandb.ai/tyranguyen7-danmarks-tekniske-universitet-dtu/football-lstm`n"
Write-Host "================================================================================`n" -ForegroundColor Green
