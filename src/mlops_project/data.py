from pathlib import Path
import typer
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MyDataset(Dataset):
    """Loading preprocessed historical football match data."""

    def __init__(self, data_path: Path, seq_len: int = 10) -> None:
        self.data_path = data_path
        self.seq_len = seq_len
        self.df = pd.read_csv(data_path)

        # Encoding target labels
        self.label_map = {"home": 0, "draw": 1, "away": 2}
        self.y = self.df["target"].map(self.label_map).values.astype("int64")

        # Dropping non-feature data not to be fed into model (i.e., redundant)
        drop_cols = [
            "id",
            "target",
            "home_team_name",
            "away_team_name",
            "league_name",
            "match_date"
        ]
        self.df = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns])

        # Grouping features per historical match
        self.seq_columns = []
        for i in range(1, self.seq_len + 1):
            cols_i = [c for c in self.df.columns if c.endswith(f"_{i}")]
            self.seq_columns.append(cols_i)

        # Input size per match
        self.input_size = len(self.seq_columns[0])

        self.df = self.df.fillna(0)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index: int):
        """
        Return a given sample from the dataset.
        Shape: (seq_len, features_per_match)
        """
        row = self.df.iloc[index]

        # Grouping columns by historical match number
        seq = []
        for cols in self.seq_columns:
            match_features = row[cols].values.astype("float32")
            seq.append(match_features)
        
        x = torch.tensor(seq, dtype=torch.float32) # shape: (seq_len, features_per_match)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        df = pd.read_csv(self.data_path)
        df['match_date'] = pd.to_datetime(df['match_date'], errors="coerce")

        numeric_cols = []
        bool_cols = []
        cat_cols = []

        # =====================
        # Home team match history
        # =====================
        for i in range (1,11):
            numeric_cols += [
                f"home_team_history_goal_{i}",
                f"home_team_history_opponent_goal_{i}",
                f"home_team_history_rating_{i}",
                f"home_team_history_opponent_rating_{i}"
            ]

            bool_cols += [
                f"home_team_history_is_play_home_{i}",
                f"home_team_history_is_cup_{i}"
            ]

            cat_cols += [
                f"home_team_history_coach_{i}",
                f"home_team_history_league_id_{i}"
            ]

            # Obtaining integer number of days instead of datetime object
            df[f"home_team_history_match_days_since_{i}"] = (
                df["match_date"] - pd.to_datetime(df[f"home_team_history_match_date_{i}"], errors="coerce")
            ).dt.days

            numeric_cols.append(f"home_team_history_match_days_since_{i}")

        # =====================
        # Away team match history
        # =====================
        for i in range(1, 11):
            numeric_cols += [
                f"away_team_history_goal_{i}",
                f"away_team_history_opponent_goal_{i}",
                f"away_team_history_rating_{i}",
                f"away_team_history_opponent_rating_{i}",
            ]
            bool_cols += [
                f"away_team_history_is_play_home_{i}",
                f"away_team_history_is_cup_{i}",
            ]
            cat_cols += [
                f"away_team_history_coach_{i}",
                f"away_team_history_league_id_{i}",
            ]
            df[f"away_team_history_match_days_since_{i}"] = (
                df["match_date"]
                - pd.to_datetime(df[f"away_team_history_match_date_{i}"], errors="coerce")
            ).dt.days
            numeric_cols.append(f"away_team_history_match_days_since_{i}")

        # Dropping raw historical match dates (redundant data)
        df = df.drop(columns=[c for c in df.columns if "history_match_date" in c])

        # Encoding categorical features to integers
        for col in cat_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

        # Scaling numerical features for consistency
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Filling in NaN values
        df = df.fillna(0)

        # Save preprocessed data
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / "processed_data.csv"
        df.to_csv(output_file, index=False)

        print(f"Preprocessed data saved to {output_file}")

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
