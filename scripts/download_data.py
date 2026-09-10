"""
scripts/download_data.py - One-time dataset download and filtering script.

Downloads the ThoughtVector 'Customer Support on Twitter' dataset via kagglehub,
filters rows associated with 'AmazonHelp' (both brand replies and inbound customer queries),
and saves data/amazonhelp_raw.csv.

Requires Kaggle API credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY).
"""

import logging
import os
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_DATASET = "thoughtvector/customer-support-on-twitter"
TARGET_BRAND = "AmazonHelp"
OUTPUT_FILE = Path("data/amazonhelp_raw.csv")


def download_and_filter():
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub is not installed. Please run: pip install kagglehub")
        return

    logger.info(f"Downloading dataset '{TARGET_DATASET}' via kagglehub...")
    dataset_dir = kagglehub.dataset_download(TARGET_DATASET)
    logger.info(f"Dataset downloaded to path: {dataset_dir}")

    # Locate CSV file (typically twcs.csv or twcs/twcs.csv)
    dataset_path = Path(dataset_dir)
    csv_files = list(dataset_path.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded directory {dataset_dir}")

    main_csv = csv_files[0]
    logger.info(f"Loading main CSV from {main_csv}...")

    # Load with low_memory=False
    df = pd.read_csv(main_csv, low_memory=False)
    logger.info(f"Raw dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Clean IDs as strings to preserve precision
    df["tweet_id"] = df["tweet_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    df["author_id"] = df["author_id"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    logger.info(f"Filtering rows for brand: {TARGET_BRAND}...")
    # 1. Outbound replies authored by brand
    brand_outbound = df["author_id"].str.lower() == TARGET_BRAND.lower()

    # 2. Inbound tweets that the brand replied to
    brand_parent_ids = set(df[brand_outbound]["in_response_to_tweet_id"].unique())
    brand_parent_ids.discard("")
    brand_parent_ids.discard("nan")
    is_parent_of_reply = df["tweet_id"].isin(brand_parent_ids)

    # 3. Customer tweets explicitly mentioning the brand
    is_brand_mention = df["text"].str.contains(f"@{TARGET_BRAND}", case=False, na=False)

    filtered_mask = brand_outbound | is_parent_of_reply | is_brand_mention
    brand_df = df[filtered_mask].copy()

    logger.info(f"Filtered to {len(brand_df)} rows relevant to {TARGET_BRAND}.")
    logger.info(f"Author breakdown:\n{brand_df['author_id'].value_counts().head(5)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    brand_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved filtered dataset to {OUTPUT_FILE} (size: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    download_and_filter()
