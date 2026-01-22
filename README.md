# Machine Learning Operations - Final Project

## Project Description:

As football enthusiasts, this project aims to shift the perspective from simply being fans to using match data to gain insights into what drives a team's success. We aim to predict the probability of each match outcome (home win, draw, or away win) for each team and each match. We are using the [Football Match Probability Prediction](https://www.kaggle.com/competitions/football-match-probability-prediction/data) dataset from Kaggle. The dataset is composed of more than 150,000 football matches worldwide, with descriptive features including home vs. away teams, league, and coaches, in addition to historical information on the previous 10 matches for both home and away teams, such as date, number of goals scored, number of goals conceded, and more.

We employ a Long Short-Term Memory (LSTM) Model to leverage the sequential nature of historical match data, processing up to 10 previous matches per team to highlight temporal patterns and performance trends. The project implements a complete MLOps pipeline including CI/CD automation, cloud training on GCP Vertex AI, experiment tracking with Weights & Biases, and deployment to Google Cloud Run with comprehensive monitoring (Prometheus, Evidently drift detection). Additionally, we developed a Streamlit frontend for user-friendly predictions.

## Project structure

The directory structure is as follows:
```txt
├── .github/                        # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yaml                 # CI tests (Ubuntu/macOS/Windows)
│       ├── docker.yaml             # Docker build pipeline
│       ├── data_changes.yaml       # Trigger on DVC data changes
│       └── model_changes.yaml      # Trigger on model registry changes
├── configs/                        # Hydra configuration files
│   ├── train.yaml                  # Main training config
│   ├── train_gcp.yaml              # GCP training config
│   ├── train_gpu.yaml              # GPU training config
│   ├── train_sample.yaml           # Sample training config
│   └── sweep.yaml                  # W&B sweep config
├── data/                           # Data directory (DVC tracked)
│   ├── frontend/                   # Frontend assets
│   │   ├── league_mapping.json
│   │   ├── league_id_to_encoded.json
│   │   └── example_games_real.json
│   ├── processed/
│   │   └── processed_data.csv
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── train_target_and_scores.csv
│   ├── raw.dvc                     # DVC pointer file
│   └── README.md
├── models/                         # Trained models
│   └── best_model.pth              # PyTorch model checkpoint
├── notebooks/                      # Jupyter notebooks for exploration
├── reports/                        # Reports and analysis
│   ├── figures/
│   ├── report.py                   # Report generation script
│   └── README.md
├── scripts/                        # Utility scripts
│   ├── create_balanced_sample.py
│   ├── create_league_encoding_mapping.py
│   ├── create_league_mapping.py
│   ├── deploy_api.py
│   ├── extract_real_examples.py
│   ├── launch_sweep.py
│   ├── run_sweep.bat
│   ├── run_sweep.py
│   ├── run_sweeps_venv.ps1
│   ├── setup_monitoring.py
│   ├── submit_gcp_gpu.py
│   ├── submit_gcp_training.py
│   ├── test_drift_monitoring.py
│   └── wandb_init.py
├── src/                            # Source code
│   └── mlops_project/
│       ├── __init__.py
│       ├── api.py                  # FastAPI application
│       ├── data.py                 # Dataset class
│       ├── data_drift.py           # Evidently AI drift detection
│       ├── evaluate.py             # Model evaluation
│       ├── export_onnx.py          # ONNX export utility
│       ├── model.py                # LSTM model with attention
│       ├── profile.py              # cProfile performance analysis
│       ├── test_torch_compile.py   # torch.compile testing
│       ├── train.py                # Training script (Hydra)
│       ├── train_sweep.py          # W&B sweep training
│       └── visualize.py            # Visualization utilities
├── tests/                          # Test suite (17 tests)
│   ├── __init__.py
│   ├── test_api.py                 # API endpoint tests (7 tests)
│   ├── test_data.py                # Dataset tests (5 tests)
│   ├── test_model.py               # Model architecture tests (5 tests)
│   └── locustfile.py               # Load testing with Locust
├── wandb/                          # Weights & Biases artifacts
├── .dvc/                           # DVC configuration
├── .dvcignore                      # DVC ignore patterns
├── .gitignore                      # Git ignore patterns
├── .pre-commit-config.yaml         # Pre-commit hooks (ruff)
├── .python-version                 # Python version specification
├── api.dockerfile                  # API Docker image
├── cloudbuild.yaml                 # GCP Cloud Build configuration
├── frontend.py                     # Streamlit frontend
├── LICENSE                         # Project license
├── pyproject.toml                  # Python project configuration
├── README.md                       # Project README
├── requirements.txt                # Project dependencies
├── requirements_dev.txt            # Development dependencies
├── tasks.py                        # Invoke automation tasks
├── test_drift_monitoring.py        # Drift monitoring test script
└── train.dockerfile                # Training Docker image
```
