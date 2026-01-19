# Data Management

## DVC Configuration

This project uses DVC (Data Version Control) to track data versions. The DVC configuration is in place, but due to Google OAuth/service account limitations, data must be downloaded manually.

## Getting the Data

**Option 1: Manual Download (Recommended)**
1. Download raw data from: https://drive.google.com/drive/folders/1kau2upzOnWRSpUZndTkGygA5hepRRYNK?usp=sharing
2. Place files in `data/raw/`:
   - `train.csv`
   - `test.csv`
   - `train_target_and_scores.csv`
3. Run preprocessing: `invoke preprocess-data`

**Option 2: DVC Pull (If you have access)**
```bash
dvc pull
```

Note: DVC remote is configured to Google Drive but requires proper authentication setup. For team collaboration, manual download from the shared folder is recommended.

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

## DVC Status

- ✅ DVC is configured and tracking data versions
- ✅ `.dvc` files are committed to Git
- ⚠️ Remote storage: Google Drive (manual access required)
- ✅ Data versioning: Tracked in `data.dvc`
