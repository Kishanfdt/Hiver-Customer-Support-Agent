"""
src/retrieval.py - Precedent resolution retriever using TF-IDF sparse matching.

Includes retrieval data-leakage protection (exclude_ids) to prevent held-out
test or golden set items from surfacing during evaluation.
"""

from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ResolutionRetriever:
    """
    Retrieves the top-k most similar past customer complaints and their official
    brand resolution replies.
    """

    def __init__(self, max_features: int = 15000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=max_features,
            sublinear_tf=True
        )
        self.corpus_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.is_fitted = False

    def fit(self, pairs_df: pd.DataFrame) -> "ResolutionRetriever":
        """
        Fits TF-IDF vectorizer on the paired complaint texts.
        Expected DataFrame columns: 'tweet_id', 'text', 'resolution_text'
        """
        if pairs_df.empty or "text" not in pairs_df.columns:
            self.corpus_df = pd.DataFrame(columns=["tweet_id", "text", "resolution_text"])
            self.is_fitted = False
            return self

        self.corpus_df = pairs_df.copy().reset_index(drop=True)
        # Ensure tweet_id is string
        self.corpus_df["tweet_id"] = self.corpus_df["tweet_id"].astype(str)
        texts = self.corpus_df["text"].fillna("").astype(str).tolist()

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return self

    def top_k(
        self,
        query: str,
        k: int = 3,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most similar precedents for the query.

        Args:
            query: The customer complaint text.
            k: Maximum number of precedents to return.
            exclude_ids: Optional set of tweet_ids to exclude from candidate matches
                         (crucial for preventing eval retrieval data leakage).

        Returns:
            List of dicts: [{"tweet_id": str, "text": str, "resolution_text": str, "similarity": float}]
        """
        if not self.is_fitted or self.corpus_df is None or len(self.corpus_df) == 0:
            return []

        if not query or not query.strip():
            return []

        exclude_set = {str(i) for i in exclude_ids} if exclude_ids else set()

        query_vec = self.vectorizer.transform([query])
        # Compute cosine similarity
        similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()

        # Rank indices descending
        sorted_indices = np.argsort(-similarities)

        results = []
        for idx in sorted_indices:
            tweet_id = str(self.corpus_df.iloc[idx]["tweet_id"])
            if tweet_id in exclude_set:
                continue

            score = float(similarities[idx])
            results.append({
                "tweet_id": tweet_id,
                "text": str(self.corpus_df.iloc[idx]["text"]),
                "resolution_text": str(self.corpus_df.iloc[idx]["resolution_text"]),
                "similarity": round(score, 4)
            })

            if len(results) >= k:
                break

        return results
