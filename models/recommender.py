"""
Hybrid Movie Recommendation Engine
Combines:
  - Content-Based Filtering (TF-IDF on genres + title)
  - Collaborative Filtering (SVD via Truncated SVD on user-item matrix)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hybrid_model.pkl")


class HybridRecommender:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.movie_index = None   # title → index
        self.index_movie = None   # index → movie_id

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        """Train both content-based and collaborative models."""
        self.movies = movies.reset_index(drop=True)
        self.ratings = ratings

        # --- index maps ---
        self.movie_index = {row["title"]: idx for idx, row in self.movies.iterrows()}
        self.index_movie = {idx: row["movie_id"] for idx, row in self.movies.iterrows()}

        self._fit_content()
        self._fit_collaborative()

    def _fit_content(self):
        """TF-IDF on genres + title text."""
        self.movies["content"] = (
            self.movies["genres"].str.replace("|", " ", regex=False)
            + " " + self.movies["title"]
        )
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies["content"])

    def _fit_collaborative(self, n_components: int = 50):
        """SVD on sparse user-item rating matrix."""
        n_users = self.ratings["user_id"].max() + 1
        n_items = self.movies["movie_id"].max() + 1

        rows = self.ratings["user_id"].values
        cols = self.ratings["movie_id"].values
        data = self.ratings["rating"].values.astype(np.float32)

        self.user_item_matrix = csr_matrix((data, (rows, cols)),
                                           shape=(n_users, n_items))
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T   # shape: (n_items, n_components)

    # ------------------------------------------------------------------ #
    #  Recommendation                                                      #
    # ------------------------------------------------------------------ #

    def content_based(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        """Return top-N movies similar to the given title using TF-IDF cosine sim."""
        if movie_title not in self.movie_index:
            return pd.DataFrame()
        idx = self.movie_index[movie_title]
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_scores[idx] = 0  # exclude itself
        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        result = self.movies.iloc[top_indices][["movie_id", "title", "genres", "year"]].copy()
        result["score"] = sim_scores[top_indices]
        return result.reset_index(drop=True)

    def collaborative(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        """Return top-N movies via SVD item-item similarity."""
        if movie_title not in self.movie_index:
            return pd.DataFrame()
        idx = self.movie_index[movie_title]
        movie_id = self.index_movie[idx]

        if movie_id >= self.item_factors.shape[0]:
            return pd.DataFrame()

        item_vec = self.item_factors[movie_id].reshape(1, -1)
        sim_scores = cosine_similarity(item_vec, self.item_factors).flatten()
        sim_scores[movie_id] = 0

        # Map movie_id → dataframe index
        id_to_idx = {row["movie_id"]: i for i, row in self.movies.iterrows()}
        top_ids = np.argsort(sim_scores)[::-1][:top_n * 2]
        rows = []
        for mid in top_ids:
            if mid in id_to_idx:
                rows.append(id_to_idx[mid])
            if len(rows) >= top_n:
                break

        result = self.movies.iloc[rows][["movie_id", "title", "genres", "year"]].copy()
        result["score"] = [sim_scores[self.index_movie[r]] for r in rows]
        return result.reset_index(drop=True)

    def hybrid(self, movie_title: str, top_n: int = 10,
               alpha: float = 0.5) -> pd.DataFrame:
        """
        Hybrid: weighted blend of content + collaborative scores.
        alpha=0.5 means equal weight; higher alpha = more collaborative.
        """
        cb = self.content_based(movie_title, top_n=top_n * 2)
        cf = self.collaborative(movie_title, top_n=top_n * 2)

        if cb.empty and cf.empty:
            return pd.DataFrame()
        if cb.empty:
            return cf.head(top_n)
        if cf.empty:
            return cb.head(top_n)

        cb = cb.rename(columns={"score": "cb_score"})
        cf = cf.rename(columns={"score": "cf_score"})

        merged = pd.merge(cb, cf[["movie_id", "cf_score"]],
                          on="movie_id", how="outer")
        merged["cb_score"] = merged["cb_score"].fillna(0)
        merged["cf_score"] = merged["cf_score"].fillna(0)

        # Normalize both scores to [0, 1]
        for col in ["cb_score", "cf_score"]:
            max_val = merged[col].max()
            if max_val > 0:
                merged[col] = merged[col] / max_val

        merged["hybrid_score"] = (1 - alpha) * merged["cb_score"] + alpha * merged["cf_score"]
        merged = merged.sort_values("hybrid_score", ascending=False).head(top_n)
        merged = merged.rename(columns={"hybrid_score": "score"})
        return merged[["movie_id", "title", "genres", "year", "score"]].reset_index(drop=True)

    def filter_by_genre(self, results: pd.DataFrame, genre: str) -> pd.DataFrame:
        """Filter recommendation results by genre."""
        if not genre or genre == "All":
            return results
        return results[results["genres"].str.contains(genre, case=False, na=False)].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path: str = MODEL_PATH) -> "HybridRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def exists(path: str = MODEL_PATH) -> bool:
        return os.path.exists(path)
