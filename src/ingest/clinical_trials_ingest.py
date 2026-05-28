"""
clinical_trials_ingest.py — Pull trial data from ClinicalTrials.gov

Every clinical trial on humans must be registered in a public government
database. This script finds the trial behind each FDA approval and pulls
key details like phase and enrollment size which is two of our strongest features
for predicting how much a stock will move.
"""

import requests
import pandas as pd
import time
from pathlib import Path
from loguru import logger


# ── File paths ─────────────────────────────────────────────────────────────────
fda_data_file = Path("data/raw/fda_approvals.csv")
output_file = Path("data/raw/clinical_trials.csv")

# ClinicalTrials.gov API
ct_api_url = "https://clinicaltrials.gov/api/v2/studies"


def search_trial(drug_name):
    """
    Search ClinicalTrials.gov for a completed trial matching the drug name.
    Returns the most relevant trial's details, or None if nothing found.
    """
    if not drug_name or str(drug_name).strip() == "" or str(drug_name) == "nan":
        return None

    search_params = {
        "query.intr": drug_name,           # Search by drug/intervention name
        "filter.overallStatus": "COMPLETED", # Only completed trials
        "fields": "NCTId,Phase,EnrollmentCount,BriefTitle,PrimaryOutcomeMeasure",
        "pageSize": 3,                      # Just grab the top 3 matches
        "format": "json",
    }

    try:
        response = requests.get(ct_api_url, params=search_params, timeout=30)

        if response.status_code != 200:
            return None

        studies = response.json().get("studies", [])

        if not studies:
            return None

        # Parse each study and pick the best one (highest phase)
        parsed = [parse_study(s) for s in studies]
        parsed = [p for p in parsed if p is not None]

        if not parsed:
            return None

        # Sort by phase number and the highest phase is most relevant
        parsed.sort(key=lambda x: x["phase_num"], reverse=True)
        return parsed[0]

    except Exception as e:
        logger.debug(f"Search failed for {drug_name}: {e}")
        return None


def parse_study(study):
    """
    Extract the fields we need from a single ClinicalTrials.gov study record.
    """
    try:
        protocol = study.get("protocolSection", {})

        # Study ID (e.g. "NCT00012345")
        nct_id = protocol.get("identificationModule", {}).get("nctId", "")

        # Trial title
        title = protocol.get("identificationModule", {}).get("briefTitle", "")

        # Phase — this is our most important feature
        # Phase 3 = final large trial before approval, biggest market impact
        # Phase 2 = mid-stage, Phase 1 = early safety only
        phases = protocol.get("designModule", {}).get("phases", [])
        phase_str = phases[0] if phases else "UNKNOWN"

        # Convert phase string to a number the model can use
        phase_lookup = {
            "PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4,
            "NA": 0, "UNKNOWN": 0,
        }
        phase_num = phase_lookup.get(phase_str.replace(" ", "").upper(), 0)

        # Enrollment = how many patients were in the trial
        # Larger enrollment = more robust data = bigger market reaction
        enrollment_info = protocol.get("designModule", {}).get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count", 0) or 0

        # Primary endpoint = what the trial was trying to prove
        # e.g. "Overall Survival", "Progression-Free Survival"
        outcomes = protocol.get("outcomesModule", {}).get("primaryOutcomes", [])
        primary_endpoint = outcomes[0].get("measure", "") if outcomes else ""

        return {
            "nct_id": nct_id,
            "trial_title": title,
            "phase": phase_str,
            "phase_num": phase_num,
            "enrollment": enrollment,
            "primary_endpoint": primary_endpoint,
            "trial_found": True,
        }

    except Exception as e:
        logger.debug(f"Failed to parse study: {e}")
        return None


def run():
    if not fda_data_file.exists():
        logger.error("Run fda_ingest.py first.")
        return

    approvals = pd.read_csv(fda_data_file, parse_dates=["approval_date"])

    # Only search for recent approvals
    approvals = approvals[approvals["approval_date"].dt.year >= 2015]
    logger.info(f"Searching ClinicalTrials.gov for {len(approvals)} drugs")

    all_results = []

    for idx, row in approvals.iterrows():
        drug_name = str(row.get("generic_name", ""))
        app_number = row["application_number"]

        logger.info(f"[{idx+1}/{len(approvals)}] Searching for: {drug_name}")

        trial = search_trial(drug_name)

        if trial:
            logger.info(f"  Found: Phase {trial['phase_num']}, {trial['enrollment']} patients")
            all_results.append({
                "application_number": app_number,
                **trial,
            })
        else:
            logger.info(f"  No trial found")
            all_results.append({
                "application_number": app_number,
                "trial_found": False,
                "phase_num": 0,
                "enrollment": 0,
                "phase": "UNKNOWN",
                "primary_endpoint": "",
            })

        # Small pause between requests
        time.sleep(0.3)

    results_df = pd.DataFrame(all_results)

    # Print a summary
    found = results_df["trial_found"].sum()
    phase3_count = (results_df["phase_num"] == 3).sum()
    logger.info(f"\nResults:")
    logger.info(f"  Trials found: {found}/{len(results_df)}")
    logger.info(f"  Phase 3 trials: {phase3_count}")
    logger.info(f"  Avg enrollment: {results_df['enrollment'].mean():.0f} patients")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.success(f"Saved to {output_file}")

    print(results_df[results_df["trial_found"] == True].head(10).to_string())


if __name__ == "__main__":
    run()