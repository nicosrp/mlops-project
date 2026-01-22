# Data

## Getting the Data

The original dataset is available on Kaggle: [Soccer Match Event Dataset](https://www.kaggle.com/datasets/secareanualin/football-events)

1. Download the raw data files from Kaggle
2. Place them in `data/raw/`:
   - `train.csv`
   - `test.csv`
   - `train_target_and_scores.csv`
3. Run preprocessing: `invoke preprocess-data`

The data is also stored in Google Cloud Storage for production use with the deployed model and training pipeline.

DVC is configured for version tracking.

## Data Structure

```
data/
├── raw/           # Raw downloaded data
│   ├── train.csv
│   ├── test.csv
│   └── train_target_and_scores.csv
└── processed/     # Preprocessed data (generated)
    └── processed_data.csv
```
