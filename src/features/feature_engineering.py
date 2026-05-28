"""
feature_engineering.py — Transform raw data into model-ready features.

Raw data isn't ready for a model yet. This script:
1. Selects the columns that are actually useful for prediction
2. Scales numeric features so they're all on the same range
3. Splits the data into train and test sets by time
4. Saves everything ready for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler
import joblib


# ── File paths ─────────────────────────────────────────────────────────────────
input_file = Path("data/processed/master_dataset.csv")
output_train = Path("data/processed/train.csv")
output_test = Path("data/processed/test.csv")
scaler_file = Path("data/processed/scaler.joblib")

# ── Features we'll use to train the model ─────────────────────────────────────
# These are the numeric columns the model will learn from.
# We chose these because they have real predictive logic behind them:
#   - phase_num: Phase 3 approvals move markets more than Phase 1
#   - enrollment: bigger trials = more credible = bigger reaction
#   - is_priority_review: FDA fast-track signals urgency
#   - volatility_30d: controls for how jumpy the stock already is
#   - approval_month: seasonal patterns in market attention
#   - filing_found: whether we found an SEC filing (proxy for company size)
#   - trial_found: whether we matched a clinical trial
FEATURE_COLUMNS = [
    "phase_num",
    "enrollment",
    "is_priority_review",
    "volatility_30d",
    "approval_month",
    "approval_quarter",
    "filing_found",
    "trial_found",
]

# What we're trying to predict
TARGET_COLUMN = "pct_change_day3"      # regression: how much did it move?
DIRECTION_COLUMN = "direction"          # classification: up or down?


def load_data():
    """Load the master dataset."""
    if not input_file.exists():
        raise FileNotFoundError("Run merge_datasets.py first.")

    df = pd.read_csv(input_file, parse_dates=["approval_date"])
    logger.info(f"Loaded {len(df)} rows")
    return df


def engineer_features(df):
    """
    Create and clean all features.

    This is where we transform raw columns into model inputs.
    """
    df = df.copy()

    # Convert boolean columns to 0/1 integers
    # Models need numbers, not True/False
    df["filing_found"] = df["filing_found"].astype(int)
    df["trial_found"] = df["trial_found"].astype(int)
    df["is_priority_review"] = df["is_priority_review"].astype(int)

    # Log-scale enrollment — trial sizes range from 10 to 10,000+
    # Without scaling, large trials would dominate the model
    # log(1000) = 6.9, log(100) = 4.6 — much more balanced
    df["enrollment"] = np.log1p(df["enrollment"])

    # Fill any remaining missing values with 0
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    logger.info(f"Feature columns: {FEATURE_COLUMNS}")
    logger.info(f"Feature matrix shape: {df[FEATURE_COLUMNS].shape}")

    return df


def split_by_time(df):
    """
    Split into train and test sets based on time.

    IMPORTANT: We never use a random split for time-series data.
    If we randomly split, the model could train on a 2024 event
    and test on a 2015 event — that's data leakage (cheating).

    Instead we train on older data and test on newer data,
    which is how the model would actually work in production.

    Train: 2010 - 2022
    Test:  2023 - present
    """
    train = df[df["approval_date"].dt.year <= 2022].copy()
    test = df[df["approval_date"].dt.year >= 2023].copy()

    logger.info(f"Train set: {len(train)} rows ({train['approval_date'].dt.year.min()} - 2022)")
    logger.info(f"Test set:  {len(test)} rows (2023 - present)")

    # Print class balance
    for name, split in [("Train", train), ("Test", test)]:
        if len(split) > 0:
            up_pct = split[DIRECTION_COLUMN].mean() * 100
            logger.info(f"  {name} — UP: {up_pct:.0f}%, DOWN: {100-up_pct:.0f}%")

    return train, test


def scale_features(train, test):
    """
    Scale all numeric features to have mean=0 and std=1.

    Why scaling matters:
        volatility_30d might be 0.01 - 0.10
        enrollment (log-scaled) might be 0 - 9
        Without scaling, enrollment dominates just because it's bigger

    CRITICAL: We fit the scaler on training data only, then apply
    it to test data. Never fit on test data — that would be leakage.
    """
    scaler = StandardScaler()

    # Learn the scaling parameters from training data
    train[FEATURE_COLUMNS] = scaler.fit_transform(train[FEATURE_COLUMNS])

    # Apply the same scaling to test data (don't refit!)
    test[FEATURE_COLUMNS] = scaler.transform(test[FEATURE_COLUMNS])

    # Save the scaler so we can apply it to new data later
    joblib.dump(scaler, scaler_file)
    logger.info(f"Scaler saved to {scaler_file}")

    return train, test, scaler


def run():
    # Load data
    df = load_data()

    # Engineer features
    df = engineer_features(df)

    # Time-based split
    train, test = split_by_time(df)

    if len(train) == 0 or len(test) == 0:
        logger.error("Not enough data for train/test split.")
        return

    # Scale features
    train, test, scaler = scale_features(train, test)

    # Save splits
    output_train.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_train, index=False)
    test.to_csv(output_test, index=False)

    logger.success(f"Saved train set: {len(train)} rows → {output_train}")
    logger.success(f"Saved test set:  {len(test)} rows → {output_test}")

    # Print a preview of the features
    logger.info("\nFeature preview (train set):")
    print(train[FEATURE_COLUMNS + [TARGET_COLUMN, DIRECTION_COLUMN]].head(10).to_string())


if __name__ == "__main__":
    run()