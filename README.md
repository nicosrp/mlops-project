# Machine Learning Operations - Final Project

## Project Description:

As football enthusiasts, this project aims to shift the perspective from simply being fans to using match data to gain insights into what drives a team's success. We aim to predict the probability of each match outcome (home win, draw, or away win) for each team and each match. We are using the [Football Match Probability Prediction](https://www.kaggle.com/competitions/football-match-probability-prediction/data) dataset from the Kaggle competition. The dataset is composed of more than 150,000 football matches in the world, with descriptive features on home vs. away teams, league, and coaches, in addition to historical information on the previous 10 matches for both home and away teams, such as date, number of goals scored, number of goals conceded, and more.

A potential pipeline could include training a multiclass probabilistic classifier with cross-entropy loss, tracking experiments and model versions using Weights and Biases, and deploying an ML service that predicts home, draw, and away win probabilities for incoming match data. Results could be benchmarked against dummy-like equal probability classifications or the previous winners' choice. We may also explore an LSTM due to the sequence-nature of the data.

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
│   ├── best_model.pth              # PyTorch model
│   └── best_model.onnx             # ONNX optimized model
├── notebooks/                      # Jupyter notebooks
├── reports/                        # Reports and analysis
│   ├── figures/
│   └── Report.md                   # Exam report
├── scripts/                        # Utility scripts
│   ├── create_league_mapping.py
│   ├── create_league_encoding_mapping.py
│   ├── extract_real_examples.py
│   ├── test_drift_monitoring.py
│   ├── setup_monitoring.py
│   ├── deploy_api.py
│   ├── launch_sweep.py
│   ├── submit_gcp_training.py
│   └── wandb_init.py
├── src/                            # Source code
│   └── mlops_project/
│       ├── __init__.py
│       ├── api.py                  # FastAPI with ONNX/PyTorch support
│       ├── data.py                 # Dataset class
│       ├── data_drift.py           # Evidently AI drift detection
│       ├── evaluate.py             # Model evaluation
│       ├── model.py                # LSTM model with attention
│       ├── export_onnx.py          # ONNX export utility
│       ├── profiling.py            # cProfile performance analysis
│       ├── test_torch_compile.py   # torch.compile testing
│       ├── train.py                # Training script (Hydra)
│       ├── train_sweep.py          # W&B sweep training
│       └── visualize.py            # Visualization utilities
├── tests/                          # Test suite (28 tests, 21% coverage)
│   ├── __init__.py
│   ├── test_api.py                 # API endpoint tests
│   ├── test_data.py                # Dataset tests
│   ├── test_model.py               # Model architecture tests
│   └── locustfile.py               # Load testing
├── wandb/                          # Weights & Biases artifacts
├── .dvc/                           # DVC configuration
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml         # Pre-commit hooks (ruff, yaml)
├── LICENSE
├── api.dockerfile                  # API Docker image (ONNX)
├── train.dockerfile                # Training Docker image
├── cloudbuild.yaml                 # GCP Cloud Build trigger
├── frontend.py                     # Streamlit frontend
├── pyproject.toml                  # Python project file
├── README.md                       # Project README
├── requirements.txt                # Project dependencies
├── requirements_dev.txt            # Development requirements
└── tasks.py                        # Invoke tasks
```
