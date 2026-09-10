"""
scripts/download_data.py - One-time dataset download and filtering script.

Downloads the ThoughtVector 'Customer Support on Twitter' dataset via kagglehub,
filters rows associated with 'AmazonHelp' (both brand replies and inbound customer queries),
and saves data/amazonhelp_raw.csv under the 50MB GitHub file limit.
"""

import logging
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

    # Locate main dataset CSV file by largest file size (twcs.csv, not sample.csv)
    dataset_path = Path(dataset_dir)
    csv_files = list(dataset_path.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded directory {dataset_dir}")

    main_csv = max(csv_files, key=lambda p: p.stat().st_size)
    logger.info(f"Loading main CSV from {main_csv} (size: {main_csv.stat().st_size / (1024*1024):.2f} MB)...")

    df = pd.read_csv(main_csv, low_memory=False)
    logger.info(f"Raw dataset shape: {df.shape}")

    # Clean IDs as strings and strip trailing .0 to ensure matching
    df["tweet_id"] = df["tweet_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    df["author_id"] = df["author_id"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    logger.info(f"Filtering rows for brand: {TARGET_BRAND}...")
    brand_outbound = df["author_id"].str.lower() == TARGET_BRAND.lower()
    
    # Sample up to 75,000 brand replies and match their parent inbound tweets
    # to keep committed CSV file size under 30MB (well below GitHub's 50MB threshold)
    replies_sample = df[brand_outbound].head(75000)
    parent_ids_sample = set(replies_sample["in_response_to_tweet_id"].unique())
    parent_ids_sample.discard("")
    parent_ids_sample.discard("nan")
    
    parents_sample = df[df["tweet_id"].isin(parent_ids_sample)]
    combined_df = pd.concat([parents_sample, replies_sample]).drop_duplicates(subset=["tweet_id"])

    logger.info(f"Filtered to {len(combined_df)} direct interaction rows ({len(parents_sample)} customer queries, {len(replies_sample)} brand replies).")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved filtered dataset to {OUTPUT_FILE} (size: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    download_and_filter()
