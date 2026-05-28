"""
merge_datasets.py — Combine all 4 datasets into one master table.

We have 4 separate CSV files:
    1. fda_approvals.csv   — the drug approval decisions
    2. stock_prices.csv    — the price movements after each approval
    3. sec_filings.csv     — the 8-K filing text
    4. clinical_trials.csv — the trial phase and enrollment data

This script joins them all together using the application_number
as the common link, like stapling 4 spreadsheets into one.
"""

import pandas as pd
from pathlib import Path
from loguru import logger


# File paths 
fda_file = Path("data/raw/fda_approvals.csv")
stock_file = Path("data/raw/stock_prices.csv")
filings_file = Path("data/raw/sec_filings.csv")
trials_file = Path("data/raw/clinical_trials.csv")
output_file = Path("data/processed/master_dataset.csv")


def run():
    # Load all 4 datasets 
    logger.info("Loading datasets...")

    fda_df = pd.read_csv(fda_file, parse_dates=["approval_date"])
    stock_df = pd.read_csv(stock_file, parse_dates=["approval_date"])
    filings_df = pd.read_csv(filings_file, parse_dates=["approval_date"])
    trials_df = pd.read_csv(trials_file)

    logger.info(f"  FDA approvals:    {len(fda_df)} rows")
    logger.info(f"  Stock prices:     {len(stock_df)} rows")
    logger.info(f"  SEC filings:      {len(filings_df)} rows")
    logger.info(f"  Clinical trials:  {len(trials_df)} rows")

    # Start with stock prices as the base 
        # We only care about approval events where we have stock price data,
        # because pct_change_day3 is our target variable — what we're predicting.
        # Without it, a row is useless for training.
    logger.info("\nMerging datasets...")

    master = stock_df.copy()

    # Add FDA approval details 
        # Bring in brand name, generic name, and priority review flag
    fda_cols = ["application_number", "generic_name", "is_priority_review"]
    master = master.merge(
        fda_df[fda_cols],
        on="application_number",
        how="left"  # keep all stock rows, add FDA info where available
    )

    # Add SEC filing text
    filings_cols = ["application_number", "filing_found", "filing_text"]
    master = master.merge(
        filings_df[filings_cols],
        on="application_number",
        how="left"
    )

    # Add clinical trial data 
    trials_cols = ["application_number", "phase_num", "enrollment",
                   "phase", "primary_endpoint", "trial_found"]
    master = master.merge(
        trials_df[trials_cols],
        on="application_number",
        how="left"
    )

    # Clean up missing values
        # Fill missing text with empty string
    master["filing_text"] = master["filing_text"].fillna("")
    master["primary_endpoint"] = master["primary_endpoint"].fillna("")

        # Fill missing numbers with 0
    master["phase_num"] = master["phase_num"].fillna(0).astype(int)
    master["enrollment"] = master["enrollment"].fillna(0).astype(int)
    master["is_priority_review"] = master["is_priority_review"].fillna(False).astype(int)

        # Fill missing booleans with False
    master["filing_found"] = master["filing_found"].fillna(False)
    master["trial_found"] = master["trial_found"].fillna(False)

        # Add time features 
        # The month and quarter of an approval can affect market reaction
        # (e.g. Q4 approvals sometimes get more analyst attention)
    master["approval_year"] = master["approval_date"].dt.year
    master["approval_month"] = master["approval_date"].dt.month
    master["approval_quarter"] = master["approval_date"].dt.quarter

    # Summary 
    logger.info(f"\nMaster dataset summary:")
    logger.info(f"  Total rows: {len(master)}")
    logger.info(f"  Columns: {list(master.columns)}")
    logger.info(f"  Stocks went UP:   {(master['direction'] == 1).sum()}")
    logger.info(f"  Stocks went DOWN: {(master['direction'] == 0).sum()}")
    logger.info(f"  Has SEC filing:   {master['filing_found'].sum()}")
    logger.info(f"  Has trial data:   {master['trial_found'].sum()}")
    logger.info(f"  Avg phase:        {master['phase_num'].mean():.1f}")
    logger.info(f"  Avg enrollment:   {master['enrollment'].mean():.0f}")

    # ── Step 9: Save ───────────────────────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_file, index=False)
    logger.success(f"Saved master dataset to {output_file}")

    print("\nFirst 5 rows:")
    print(master.head().to_string())


if __name__ == "__main__":
    run()