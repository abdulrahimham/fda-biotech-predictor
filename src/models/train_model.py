"""
train_model.py — Train and evaluate the prediction model.

We train two models:
1. Classification: will the stock go UP or DOWN? (predicts direction)
2. Regression: by how much will it move? (predicts magnitude)

We use Random Forest for both because:
- Works well on small datasets (we have 47 training rows)
- Handles mixed features well (some binary, some continuous)
- Easy to interpret — we can see which features matter most
- No need for GPU or complex setup like deep learning

We'll add FinBERT (the NLP model) in the next step once the
basic model is working.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
)
import joblib


# ── File paths ─────────────────────────────────────────────────────────────────
train_file = Path("data/processed/train.csv")
test_file = Path("data/processed/test.csv")
classifier_file = Path("data/processed/classifier.joblib")
regressor_file = Path("data/processed/regressor.joblib")
results_file = Path("data/processed/results.csv")

# Feature columns — must match feature_engineering.py
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

TARGET_REGRESSION = "pct_change_day3"
TARGET_CLASSIFICATION = "direction"


def load_data():
    """Load train and test sets."""
    train = pd.read_csv(train_file, parse_dates=["approval_date"])
    test = pd.read_csv(test_file, parse_dates=["approval_date"])
    logger.info(f"Train: {len(train)} rows, Test: {len(test)} rows")
    return train, test


def train_classifier(X_train, y_train):
    """
    Train a Random Forest classifier to predict UP or DOWN.

    Random Forest = many decision trees voting together.
    Each tree learns slightly different patterns from the data.
    The final prediction is the majority vote across all trees.

    n_estimators=100 means 100 trees voting together.
    random_state=42 makes results reproducible.
    """
    logger.info("Training classification model (UP vs DOWN)...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,        # limit depth to prevent overfitting on small data
        min_samples_leaf=2, # each leaf needs at least 2 samples
        random_state=42,
    )
    model.fit(X_train, y_train)
    logger.success("Classifier trained")
    return model


def train_regressor(X_train, y_train):
    """
    Train a Random Forest regressor to predict the magnitude of price change.

    Same idea as the classifier but predicts a number instead of a category.
    e.g. "stock will move +8.3%" instead of just "UP"
    """
    logger.info("Training regression model (magnitude of price change)...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    logger.success("Regressor trained")
    return model


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate the classification model on the test set.

    We compare against a naive baseline — what if we just always
    predicted UP? That's our minimum bar to beat.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Naive baseline: always predict the most common class
    most_common = y_test.mode()[0]
    baseline_preds = [most_common] * len(y_test)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    logger.info(f"\nClassification Results:")
    logger.info(f"  Model accuracy:    {accuracy*100:.1f}%")
    logger.info(f"  Naive baseline:    {baseline_accuracy*100:.1f}%")
    logger.info(f"  Improvement:       +{(accuracy-baseline_accuracy)*100:.1f}%")
    logger.info(f"\n{classification_report(y_test, predictions, target_names=['DOWN', 'UP'])}")

    return accuracy, baseline_accuracy, predictions


def evaluate_regressor(model, X_test, y_test):
    """
    Evaluate the regression model on the test set.

    MAE = Mean Absolute Error
    e.g. MAE of 5.2 means on average we're off by 5.2 percentage points
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    logger.info(f"\nRegression Results:")
    logger.info(f"  Mean Absolute Error: {mae:.2f}%")
    logger.info(f"  (On average, predictions are off by {mae:.2f} percentage points)")

    return mae, predictions


def show_feature_importance(model, feature_names):
    """
    Show which features the model found most useful.

    Feature importance tells us which inputs the model
    relied on most when making predictions.
    This is great to show in interviews.
    """
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("\nFeature Importance (what drove predictions):")
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] * 50)
        logger.info(f"  {row['feature']:<25} {bar} {row['importance']:.3f}")

    return importance


def run():
    # Load data
    train, test = load_data()

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]
    y_train_cls = train[TARGET_CLASSIFICATION]
    y_test_cls = test[TARGET_CLASSIFICATION]
    y_train_reg = train[TARGET_REGRESSION]
    y_test_reg = test[TARGET_REGRESSION]

    # ── Train models ───────────────────────────────────────────────────────────
    classifier = train_classifier(X_train, y_train_cls)
    regressor = train_regressor(X_train, y_train_reg)

    # ── Evaluate models ────────────────────────────────────────────────────────
    accuracy, baseline, cls_preds = evaluate_classifier(classifier, X_test, y_test_cls)
    mae, reg_preds = evaluate_regressor(regressor, X_test, y_test_reg)

    # ── Feature importance ─────────────────────────────────────────────────────
    logger.info("\nClassifier feature importance:")
    show_feature_importance(classifier, FEATURE_COLUMNS)

    # ── Save models ────────────────────────────────────────────────────────────
    joblib.dump(classifier, classifier_file)
    joblib.dump(regressor, regressor_file)
    logger.success(f"Models saved to {classifier_file} and {regressor_file}")

    # ── Save results for dashboard ─────────────────────────────────────────────
    results = test.copy()
    results["predicted_direction"] = cls_preds
    results["predicted_pct_change"] = reg_preds
    results["correct"] = (results["predicted_direction"] == results[TARGET_CLASSIFICATION]).astype(int)
    results.to_csv(results_file, index=False)
    logger.success(f"Results saved to {results_file}")

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Direction accuracy: {accuracy*100:.1f}% (baseline: {baseline*100:.1f}%)")
    logger.info(f"Magnitude MAE:      {mae:.2f}%")
    logger.info(f"Test set size:      {len(test)} events")


if __name__ == "__main__":
    run()