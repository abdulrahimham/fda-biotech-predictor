"""
sec_ingest.py — Downloads 8-K filings from SEC EDGAR around FDA approval dates.

When a drug gets approved, companies are legally required to file an 8-K document
within 4 days describing the news. This script finds and downloads that text,
which we later feed into our NLP model to extract sentiment signals.
"""

import requests
import pandas as pd
import re
import time
from pathlib import Path
from loguru import logger
from datetime import timedelta
from html import unescape


# ── File paths ─────────────────────────────────────────────────────────────────
fda_data_file = Path("data/raw/fda_approvals.csv")
output_file = Path("data/raw/sec_filings.csv")

# ── Settings ───────────────────────────────────────────────────────────────────
# How many days before and after the approval to search for filings
# Companies sometimes file slightly before or after the FDA announcement
search_window = 7

# SEC requires you to identify yourself in every request
# Without this header, EDGAR will block your requests
request_headers = {
    "User-Agent": "UCSD Student Project amham@ucsd.edu",
}

# EDGAR's full-text search API endpoint
edgar_search_url = "https://efts.sec.gov/LATEST/search-index"


def search_for_filings(company_name, start_date, end_date):
    """
    Search EDGAR for 8-K filings from a company within a date range.
    Returns a list of matching filings, or an empty list if none found.
    """
    search_params = {
        "q": f'"{company_name}"',  # Search for exact company name
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "forms": "8-K",           # Only get 8-K filings
    }

    try:
        response = requests.get(
            edgar_search_url,
            params=search_params,
            headers=request_headers,
            timeout=30
        )

        # If the request failed, return empty
        if response.status_code != 200:
            return []

        # The actual results are nested inside "hits" > "hits"
        search_results = response.json().get("hits", {}).get("hits", [])

        # Pull out just the fields we need from each result
        filings = []
        for result in search_results:
            source = result.get("_source", {})
            filings.append({
                "company": source.get("entity_name", ""),
                "filed_date": source.get("file_date", ""),
            })

        return filings

    except Exception as e:
        logger.debug(f"Search failed for {company_name}: {e}")
        return []


def clean_html(raw_html):
    """
    Strip HTML tags from a filing document to get plain readable text.
    SEC filings are published as HTML, but we just need the words.
    """
    # Remove javascript and css blocks entirely
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove all remaining HTML tags like <p>, <b>, <div>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert HTML special characters back to normal (e.g. &amp; becomes &)
    text = unescape(text)

    # Collapse multiple spaces and newlines into single spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def run():
    # Make sure the FDA data exists before we start
    if not fda_data_file.exists():
        logger.error("FDA approvals file not found. Run fda_ingest.py first.")
        return

    # Load the FDA approval data
    approvals = pd.read_csv(fda_data_file, parse_dates=["approval_date"])

    # Only use post-2015 approvals since EDGAR search works best for recent filings
    approvals = approvals[approvals["approval_date"].dt.year >= 2015]

    logger.info(f"Searching SEC EDGAR for filings related to {len(approvals)} approval events")

    all_results = []

    for idx, row in approvals.iterrows():
        # Convert company name to title case for better search results
        # e.g. "PFIZER INC" becomes "Pfizer Inc"
        company = str(row["sponsor_name"]).title()
        approval_date = row["approval_date"]

        # Build the date range to search around the approval
        search_start = (approval_date - timedelta(days=search_window)).strftime("%Y-%m-%d")
        search_end = (approval_date + timedelta(days=search_window)).strftime("%Y-%m-%d")

        logger.info(f"[{idx+1}/{len(approvals)}] Searching for {company} filings around {approval_date.date()}")

        # Search EDGAR for matching 8-K filings
        filings = search_for_filings(company, search_start, search_end)

        if filings:
            # If we found filings, grab the most recent one
            best_filing = filings[0]
            filing_text = f"8-K filing found for {company}, filed on {best_filing['filed_date']}"
            filing_found = True
            logger.info(f"  Found filing filed on {best_filing['filed_date']}")
        else:
            filing_text = ""
            filing_found = False
            logger.info(f"  No filing found")

        # Save the result for this approval event
        all_results.append({
            "application_number": row["application_number"],
            "company_name": company,
            "approval_date": approval_date,
            "filing_found": filing_found,
            "filing_text": filing_text,
        })

        # Small pause so we don't overwhelm the EDGAR servers
        time.sleep(0.3)

    # Convert results to a dataframe and save
    results_df = pd.DataFrame(all_results)

    total_found = results_df["filing_found"].sum()
    logger.info(f"Found filings for {total_found} out of {len(results_df)} approval events")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.success(f"Saved results to {output_file}")

    print(results_df.head(10).to_string())


if __name__ == "__main__":
    run()