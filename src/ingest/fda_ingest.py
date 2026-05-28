"""
fda_ingest.py — Download branded drug approval AND rejection decisions from the FDA.

We target NDA and BLA applications only (branded drugs, not generics).
We collect both approvals (AP) and rejections (CRL - Complete Response Letter).

    AP  = Approved
    CRL = Complete Response Letter (FDA rejection)

Including rejections doubles our dataset and makes the model more realistic —
it has to predict both stock crashes (rejections) and rallies (approvals).
"""

import requests
import pandas as pd
import time
from loguru import logger
from pathlib import Path


output_file = Path("data/raw/fda_approvals.csv")
fda_api_url = "https://api.fda.gov/drug/drugsfda.json"


def fetch_decisions(pages: int = 20):
    """
    Download both approval and rejection decisions for branded drugs.
    """
    all_records = []

    # We collect both AP (approved) and CRL (rejected)
    decision_types = ["AP", "CR"]

    for decision in decision_types:
        logger.info(f"\nFetching {decision} decisions...")

        for page in range(pages):
            skip = page * 100

            params = {
                "search": f"submissions.submission_status:{decision}",
                "limit": 100,
                "skip": skip,
            }

            logger.info(f"  Page {page + 1} of {pages}...")

            response = requests.get(fda_api_url, params=params, timeout=30)

            if response.status_code != 200:
                logger.warning(f"  API returned {response.status_code}. Moving on.")
                break

            results = response.json().get("results", [])

            if not results:
                logger.info("  No more results.")
                break

            for drug in results:
                record = parse_drug(drug, decision)
                if record:
                    all_records.append(record)

            time.sleep(0.3)

    logger.info(f"\nTotal records downloaded: {len(all_records)}")
    return pd.DataFrame(all_records)


def parse_drug(drug, decision_type):
    """
    Extract fields we need from one FDA record.
    """
    try:
        app_number = drug.get("application_number", "")

        # Skip generic drugs — we only want branded NDA and BLA
        if not (app_number.startswith("NDA") or app_number.startswith("BLA")):
            return None

        products = drug.get("products", [{}])
        first_product = products[0] if products else {}

        ingredients = first_product.get("active_ingredients", [{}])
        generic_name = ingredients[0].get("name", "") if ingredients else ""

        submissions = drug.get("submissions", [])
        decision = find_decision(submissions, decision_type)

        if not decision:
            return None

        return {
            "application_number": app_number,
            "sponsor_name": drug.get("sponsor_name", ""),
            "brand_name": first_product.get("brand_name", ""),
            "generic_name": generic_name,
            "approval_date": decision.get("submission_status_date", ""),
            "decision": decision_type,  # AP or CRL
            "is_priority_review": decision.get("review_priority", "STANDARD") != "STANDARD",
        }

    except Exception as e:
        logger.debug(f"Skipping bad record: {e}")
        return None


def find_decision(submissions, decision_type):
    """Find the submission matching the decision type (AP or CRL)."""
    for s in submissions:
        if s.get("submission_status") == decision_type:
            return s
    return None


def run():
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_decisions(pages=20)

    if df.empty:
        logger.error("No data downloaded.")
        return

    # Clean up dates
    df["approval_date"] = pd.to_datetime(
        df["approval_date"], format="%Y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["approval_date"])

    # Only keep post-2010
    df = df[df["approval_date"].dt.year >= 2010]

    df = df.sort_values("approval_date", ascending=False)
    df = df.drop_duplicates(subset=["application_number", "decision"])

    # Summary
    approvals = (df["decision"] == "AP").sum()
    rejections = (df["decision"] == "CRL").sum()
    logger.success(f"Saved {len(df)} total decisions")
    logger.info(f"  Approvals (AP):  {approvals}")
    logger.info(f"  Rejections (CRL): {rejections}")

    df.to_csv(output_file, index=False)
    print(df.head(10).to_string())


if __name__ == "__main__":
    run()