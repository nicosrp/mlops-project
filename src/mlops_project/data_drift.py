"""
Data drift monitoring using Evidently AI.

This module detects data drift, data quality issues, and target drift
in production predictions compared to training data.
"""

from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.test_suite import TestSuite


def load_reference_data(data_path: Path = Path("data/processed")) -> pd.DataFrame:
    """
    Load reference data (training data) for comparison.

    Args:
        data_path: Path to processed data directory

    Returns:
        DataFrame with reference data
    """
    # Load your training data - adjust based on your actual data format
    # This is an example - modify to match your data structure
    try:
        train_data = pd.read_csv(data_path / "train_features.csv")
        return train_data
    except FileNotFoundError:
        print(f"Warning: Training data not found at {data_path}")
        return pd.DataFrame()


def load_current_data(db_path: Path = Path("data/prediction_database.csv"), n: int | None = None) -> pd.DataFrame:
    """
    Load current production data from prediction database.

    Args:
        db_path: Path to prediction database CSV
        n: Number of latest entries to load (None = all)

    Returns:
        DataFrame with current data
    """
    if not db_path.exists():
        print(f"Warning: Prediction database not found at {db_path}")
        return pd.DataFrame()

    df = pd.read_csv(db_path)

    if n is not None:
        df = df.tail(n)

    return df


def filter_by_time(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """
    Filter dataframe to only include entries from the last N hours.

    Args:
        df: DataFrame with 'time' column
        hours: Number of hours to look back

    Returns:
        Filtered DataFrame
    """
    df["time"] = pd.to_datetime(df["time"])
    cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
    return df[df["time"] >= cutoff_time]


def standardize_dataframes(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize column names between reference and current data.

    Args:
        reference_data: Training/reference data
        current_data: Production data

    Returns:
        Tuple of (standardized_reference, standardized_current)
    """
    # Remove timestamp column from current data
    if "time" in current_data.columns:
        current_data = current_data.drop(columns=["time"])

    # Map prediction column to match training data target column
    # Adjust based on your actual column names
    if "prediction" in current_data.columns:
        # Convert prediction labels to numeric if needed
        label_map = {"home": 0, "draw": 1, "away": 2}
        current_data["target"] = current_data["prediction"].map(label_map)
        current_data = current_data.drop(columns=["prediction"])

    # Remove probability columns if present
    prob_cols = [col for col in current_data.columns if col.startswith("prob_")]
    if prob_cols:
        current_data = current_data.drop(columns=prob_cols)

    # Ensure both have the same columns
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    return reference_data, current_data


def generate_drift_report(
    reference_data: pd.DataFrame, current_data: pd.DataFrame, output_path: Path = Path("reports/data_drift_report.html")
) -> Report:
    """
    Generate Evidently data drift report.

    Args:
        reference_data: Training/reference data
        current_data: Production data
        output_path: Path to save HTML report

    Returns:
        Evidently Report object
    """
    # Define column mapping
    column_mapping = ColumnMapping(target="target", prediction=None, numerical_features=None, categorical_features=None)

    # Create report with multiple presets
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])

    # Run report
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    # Save HTML report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))

    print(f"Report saved to {output_path}")
    return report


def run_drift_tests(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> TestSuite:
    """
    Run data drift and quality tests programmatically.

    Args:
        reference_data: Training/reference data
        current_data: Production data

    Returns:
        TestSuite with results
    """
    # Define column mapping
    column_mapping = ColumnMapping(target="target", prediction=None)

    # Create test suite
    tests = TestSuite(tests=[DataDriftTestPreset(), DataQualityTestPreset()])

    # Run tests
    tests.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    # Print results
    results = tests.as_dict()
    print("\n=== Data Drift Test Results ===")
    print(f"Number of tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['success_tests']}")
    print(f"Failed: {results['summary']['failed_tests']}")

    # Print failed tests
    if results["summary"]["failed_tests"] > 0:
        print("\nFailed tests:")
        for test in results["tests"]:
            if test["status"] == "FAIL":
                print(f"  - {test['name']}: {test.get('description', 'No description')}")

    return tests


def main():
    """Main function to run data drift monitoring."""
    print("Loading reference data...")
    reference_data = load_reference_data()

    if reference_data.empty:
        print("No reference data available. Please provide training data.")
        return

    print(f"Reference data shape: {reference_data.shape}")

    print("\nLoading current production data...")
    current_data = load_current_data()

    if current_data.empty:
        print("No production data available yet. Run some predictions first.")
        return

    print(f"Current data shape: {current_data.shape}")

    print("\nStandardizing dataframes...")
    reference_data, current_data = standardize_dataframes(reference_data, current_data)

    print(f"Standardized reference shape: {reference_data.shape}")
    print(f"Standardized current shape: {current_data.shape}")

    # Generate drift report
    print("\nGenerating data drift report...")
    report = generate_drift_report(reference_data, current_data)

    # Run tests
    print("\nRunning data drift tests...")
    tests = run_drift_tests(reference_data, current_data)

    print("\n✓ Monitoring complete!")
    print("View the report at: reports/data_drift_report.html")


if __name__ == "__main__":
    main()
