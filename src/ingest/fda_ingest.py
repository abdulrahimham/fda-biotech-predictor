"""
    The FDA publishes every drug approval decision on their free API.
    This script downloads those decisions and saves them as a CSV file.
    Think of it like automatically copying a government spreadsheet to your computer.
"""

import requests
import pandas as pd
import time
from loguru import logger
from pathlib import Path

OUTPUT_PATH = Path("data/raw/fda_approvals.csv")
FDA_URL = "https://api.fda.gov/drug/drugsfda.json"


def fetch_approvals(pages: int = 5) -> pd.DataFrame:
    """
    Download drug approval records from the FDA API.
    """
    all_records = []

    for page in range(pages):
        skip = page * 100

        params = {
            "search": "submissions.submission_status:AP",
            "limit": 100,
            "skip": skip,
        }

        logger.info(f"Downloading page {page + 1} of {pages}...")

        response = requests.get(FDA_URL, params=params, timeout=30)

        if response.status_code != 200:
            logger.warning(f"API returned error {response.status_code}. Stopping.")
            break

        results = response.json().get("results", [])

        if not results:
            logger.info("No more results.")
            break

        for drug in results:
            record = parse_drug(drug)
            if record:
                all_records.append(record)

        time.sleep(0.3)

    logger.info(f"Downloaded {len(all_records)} records total")
    return pd.DataFrame(all_records)


def parse_drug(drug):
    """
    Pull out the fields we care about from one FDA record.
    """
    try:
        products = drug.get("products", [{}])
        first_product = products[0] if products else {}

        ingredients = first_product.get("active_ingredients", [{}])
        generic_name = ingredients[0].get("name", "") if ingredients else ""

        submissions = drug.get("submissions", [])
        approval = find_approval(submissions)

        if not approval:
            return None

        return {
            "application_number": drug.get("application_number", ""),
            "sponsor_name": drug.get("sponsor_name", ""),
            "brand_name": first_product.get("brand_name", ""),
            "generic_name": generic_name,
            "approval_date": approval.get("submission_status_date", ""),
            "is_priority_review": approval.get("review_priority", "STANDARD") != "STANDARD",
        }

    except Exception as e:
        logger.debug(f"Skipping bad record: {e}")
        return None


def find_approval(submissions):
    """Find the submission where the FDA said 'approved'."""
    for s in submissions:
        if s.get("submission_status") == "AP":
            return s
    return None


def run():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_approvals(pages=5)

    if df.empty:
        logger.error("No data downloaded. Check your internet connection.")
        return

    df["approval_date"] = pd.to_datetime(df["approval_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["approval_date"])
    df = df.sort_values("approval_date", ascending=False)

    df.to_csv(OUTPUT_PATH, index=False)
    logger.success(f"Saved {len(df)} records to {OUTPUT_PATH}")

    print(df.head(10).to_string())


if __name__ == "__main__":
    run()