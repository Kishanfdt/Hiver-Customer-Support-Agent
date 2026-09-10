"""
src/data_prep.py - Ingestion and dataset transformation for brand support pairs.

Loads raw customer support tweets and pairs each inbound customer query with
the target brand's official resolving response.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    """
    Load raw customer support CSV data.
    Expected columns: tweet_id, author_id, inbound, text, in_response_to_tweet_id
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    # Read CSV with string IDs to prevent precision loss on 64-bit tweet IDs
    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "inbound": str,
            "text": str,
            "in_response_to_tweet_id": str,
        },
        low_memory=False,
    )

    # Normalize inbound boolean column (can be bool, 'True'/'False', 1/0)
    if "inbound" in df.columns:
        df["inbound"] = df["inbound"].astype(str).str.strip().str.lower().isin(["true", "1", "t"])

    # Clean tweet text
    if "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str)

    # Clean in_response_to_tweet_id: strip floating decimal if read with .0
    if "in_response_to_tweet_id" in df.columns:
        df["in_response_to_tweet_id"] = (
            df["in_response_to_tweet_id"]
            .fillna("")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

    if "tweet_id" in df.columns:
        df["tweet_id"] = (
            df["tweet_id"]
            .fillna("")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

    return df


def build_brand_pairs(df: pd.DataFrame, brand: str = "AmazonHelp") -> pd.DataFrame:
    """
    Matches each inbound customer tweet to the brand's resolving reply.
    A brand resolution tweet has author_id == brand, inbound == False,
    and in_response_to_tweet_id pointing to the customer's inbound tweet_id.

    Returns DataFrame with columns: ['tweet_id', 'text', 'resolution_text']
    """
    if df.empty:
        return pd.DataFrame(columns=["tweet_id", "text", "resolution_text"])

    # Brand reply tweets: authored by brand and in response to another tweet
    brand_mask = (df["author_id"].str.lower() == brand.lower()) & (df["in_response_to_tweet_id"] != "")
    brand_replies = df[brand_mask][["in_response_to_tweet_id", "text"]].rename(
        columns={"in_response_to_tweet_id": "parent_tweet_id", "text": "resolution_text"}
    )

    # Inbound customer tweets: inbound is True or author is not the brand
    inbound_mask = df["inbound"] if "inbound" in df.columns else (df["author_id"].str.lower() != brand.lower())
    customer_tweets = df[inbound_mask][["tweet_id", "text"]]

    # Join brand resolution to inbound tweet on tweet_id == parent_tweet_id
    pairs = pd.merge(
        customer_tweets,
        brand_replies,
        left_on="tweet_id",
        right_on="parent_tweet_id",
        how="inner"
    )

    pairs = pairs[["tweet_id", "text", "resolution_text"]].drop_duplicates(subset=["tweet_id"])
    pairs = pairs[(pairs["text"].str.strip() != "") & (pairs["resolution_text"].str.strip() != "")]
    pairs.reset_index(drop=True, inplace=True)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare brand resolution pairs from raw customer support data")
    parser.add_argument("--input", "-i", type=str, default="data/amazonhelp_raw.csv", help="Path to raw CSV")
    parser.add_argument("--output", "-o", type=str, default="data/brand_pairs.csv", help="Path to output paired CSV")
    parser.add_argument("--brand", "-b", type=str, default="AmazonHelp", help="Target brand name")
    args = parser.parse_args()

    logger.info(f"Loading raw data from {args.input}...")
    df = load_raw(args.input)
    logger.info(f"Loaded {len(df)} rows. Building brand pairs for '{args.brand}'...")
    pairs = build_brand_pairs(df, brand=args.brand)
    logger.info(f"Generated {len(pairs)} matched resolution pairs.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_path, index=False)
    logger.info(f"Saved paired data to {out_path}")


if __name__ == "__main__":
    main()
