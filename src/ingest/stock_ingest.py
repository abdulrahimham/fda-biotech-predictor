"""
For each drug approval we downloaded from the FDA, we need to know
what happened to the company's stock price around that date.

This script looks up each company's stock ticker and downloads
the price history around the approval date.

The most important thing we calculate here is:
"By what percentage did the stock move in the 3 days after approval?"
That's what our model will learn to predict.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import timedelta
import time


INPUT_PATH = Path("data/raw/fda_approvals.csv")
OUTPUT_PATH = Path("data/raw/stock_prices.csv")

# Days to look at before and after the approval
DAYS_BEFORE = 30
DAYS_AFTER = 10

# Map of company name fragments to their stock ticker symbols
# e.g. if sponsor_name contains "PFIZER", the ticker is "PFE"
TICKER_MAP = {
    # Big Pharma
    "PFIZER": "PFE",
    "JOHNSON": "JNJ",
    "ABBVIE": "ABBV",
    "MERCK": "MRK",
    "LILLY": "LLY",
    "BRISTOL": "BMY",
    "AMGEN": "AMGN",
    "GILEAD": "GILD",
    "BIOGEN": "BIIB",
    "REGENERON": "REGN",
    "VERTEX": "VRTX",
    "MODERNA": "MRNA",
    "NOVARTIS": "NVS",
    "ASTRAZENECA": "AZN",
    "SANOFI": "SNY",
    "ALEXION": "ALXN",
    "INCYTE": "INCY",
    "ALNYLAM": "ALNY",
    "SEAGEN": "SGEN",
    "EXELIXIS": "EXEL",
    "JANSSEN": "JNJ",
    "GENENTECH": "RHHBY",

    # Mid-size Biotech
    "BLUEPRINT": "BPMC",
    "ACCELERON": "XLRN",
    "ARENA": "ARNA",
    "INTERCEPT": "ICPT",
    "MYOKARDIA": "MYOK",
    "GLOBAL BLOOD": "GBT",
    "DECIPHERA": "DCPH",
    "SPRING BIOSCIENCE": "SBBP",
    "COHERUS": "CHRS",
    "TRICIDA": "TCDA",
    "TURNING POINT": "TPTX",
    "IMMUNOMEDICS": "IMMU",
    "FORTY SEVEN": "FTSV",
    "PORTOLA": "PTLA",
    "DERMIRA": "DERM",
    "CORCEPT": "CORT",
    "SUPERNUS": "SUPN",
    "AIMMUNE": "AIMT",
    "MYOVANT": "MYOV",
    "KARUNA": "KRTX",
    "RHYTHM": "RYTM",
    "PRAXIS": "PRAX",
    "RELAY": "RLAY",
    "IMARA": "IMRA",
    "NUVATION": "NUVB",
    "ACADIA": "ACAD",
    "SAGE": "SAGE",
    "INTRA-CELLULAR": "ITCI",
    "NEUROCRINE": "NBIX",
    "HALOZYME": "HALO",
    "ULTRAGENYX": "RARE",
    "SAREPTA": "SRPT",
    "BIOMARIN": "BMRN",
    "IONIS": "IONS",
    "ALKERMES": "ALKS",
    "JAZZ": "JAZZ",
    "HORIZON": "HZNP",
    "PACIRA": "PCRX",
    "AVANIR": "AVNR",
    "QUESTCOR": "QCOR",
    "SUCAMPO": "SCMP",
    "DEPOMED": "DEPO",
    "KERYX": "KERX",
    "RIGEL": "RIGL",
    "CEMPRA": "CEMP",
    "TETRAPHASE": "TTPH",
    "PARATEK": "PRTK",
    "MELINTA": "MLNT",
    "NABRIVA": "NBRV",
    "CONTRAFECT": "CFRX",
    "ITERION": "ITRN",
    "ZAFGEN": "ZFGN",
    "PROTEOSTASIS": "PTI",
    "CORBUS": "CRBP",
    "CIDARA": "CDTX",
    "SCYNEXIS": "SCYX",
    "MYCOVIA": "MYCO",
    "HUMANIGEN": "HGEN",
    "TENAX": "TENX",
    "ENOCHIAN": "ENOB",
    "BIOXCEL": "BTAI",
    "AQUESTIVE": "AQST",
    "COLLEGIUM": "COLL",
    "ASSERTIO": "ASRT",
    "TREVENA": "TRVN",
    "PHIBRO": "PAHC",
    "LANNETT": "LCI",
    "AMNEAL": "AMRX",
    "HIKMA": "HIK",
    "TEVA": "TEVA",
    "MYLAN": "MYL",
    "PERRIGO": "PRGO",
    "ENDO": "ENDP",
    "MALLINCKRODT": "MNK",
    "PRESTIGE": "PBH",
    "NOVEN": "NVN",
    "NALPROPION": "NTRP",
    "OREXIGEN": "OREX",
    "VIVUS": "VVUS",
    "ARENA PHARM": "ARNA",
    "EISAI": "ESALY",
    "OTSUKA": "OTSKY",
    "DAIICHI": "DSNKY",
    "ASTELLAS": "ALPMY",
    "TAKEDA": "TAK",
    "SHIRE": "SHPG",
    "UCB": "UCBJY",
    "IPSEN": "IPSEY",
    "RECORDATI": "RCDTF",
}


def get_ticker(sponsor_name):
    """Look up the stock ticker for a company name."""
    sponsor_upper = str(sponsor_name).upper()
    for fragment, ticker in TICKER_MAP.items():
        if fragment in sponsor_upper:
            return ticker
    return None


def calculate_price_change(prices, approval_date):
    """
    Calculate how much the stock moved after the approval.

    This is the number we're trying to predict which is the percentage
    change in stock price in the 3 days after the FDA decision.

    Eample:
        Stock price day before approval: $100
        Stock price 3 days after: $120
        Price change: +20%
    """
    if prices.empty:
        return None, None, None

    prices = prices.sort_index()
    prices.index = prices.index.tz_localize(None)

    # Get the price just before the approval
    before = prices[prices.index < approval_date]
    if before.empty:
        return None, None, None
    baseline = before["Close"].iloc[-1]

    # Get prices after the approval
    after = prices[prices.index >= approval_date]
    if len(after) < 3:
        return None, None, None

    # Price 3 trading days after approval
    day3_price = after["Close"].iloc[2]

    # Percentage change formula: (new - old) / old * 100
    pct_change = (day3_price - baseline) / baseline * 100

    # Direction: did it go up or down?
    direction = 1 if pct_change > 0 else 0

    # Volatility: how much does this stock normally move day to day?
    volatility = before["Close"].pct_change().std() * 100

    return round(pct_change, 2), direction, round(volatility, 4)


def run():
    if not INPUT_PATH.exists():
        logger.error("Run fda_ingest.py first.")
        return

    approvals = pd.read_csv(INPUT_PATH, parse_dates=["approval_date"])
    logger.info(f"Processing {len(approvals)} approval events")

    # Add ticker symbols
    approvals["ticker"] = approvals["sponsor_name"].apply(get_ticker)

    # Keep only rows where we know the ticker and date is after 2000
    matched = approvals.dropna(subset=["ticker"])
    matched = matched[matched["approval_date"].dt.year >= 2000]
    logger.info(f"Matched {len(matched)} companies to stock tickers")

    results = []

    for idx, row in matched.iterrows():
        ticker = row["ticker"]
        approval_date = row["approval_date"]

        logger.info(f"[{idx+1}] Fetching {ticker} around {approval_date.date()}")

        # Download price history for a window around the approval
        start = approval_date - timedelta(days=DAYS_BEFORE + 10)
        end = approval_date + timedelta(days=DAYS_AFTER + 10)

        try:
            stock = yf.Ticker(ticker)
            prices = stock.history(start=start, end=end, auto_adjust=True)
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            continue

        if prices.empty:
            continue

        pct_change, direction, volatility = calculate_price_change(prices, approval_date)

        if pct_change is None:
            continue

        results.append({
            "application_number": row["application_number"],
            "ticker": ticker,
            "sponsor_name": row["sponsor_name"],
            "brand_name": row["brand_name"],
            "approval_date": approval_date,
            "pct_change_day3": pct_change,
            "direction": direction,
            "volatility_30d": volatility,
        })

        time.sleep(0.3)

    if not results:
        logger.error("No stock data collected.")
        return

    df = pd.DataFrame(results)

    up = (df["direction"] == 1).sum()
    down = (df["direction"] == 0).sum()
    logger.info(f"\nResults: {len(df)} events")
    logger.info(f"Went UP:   {up} ({up/len(df)*100:.0f}%)")
    logger.info(f"Went DOWN: {down} ({down/len(df)*100:.0f}%)")
    logger.info(f"Average move: {df['pct_change_day3'].mean():+.1f}%")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.success(f"Saved {len(df)} records to {OUTPUT_PATH}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    run()