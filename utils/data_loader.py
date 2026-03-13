"""
Data loader for MovieLens dataset.
Downloads and prepares the MovieLens 100K dataset automatically.
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def download_movielens():
    """Download MovieLens 100K dataset if not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ml-100k.zip")
    extract_path = os.path.join(RAW_DIR, "ml-100k")

    if not os.path.exists(extract_path):
        print("Downloading MovieLens 100K dataset...")
        response = requests.get(MOVIELENS_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)
        os.remove(zip_path)
        print("Download complete.")
    return extract_path


def load_movies():
    """Load and return movies DataFrame."""
    path = download_movielens()
    movies_file = os.path.join(path, "u.item")
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]
    cols = ["movie_id", "title", "release_date", "video_release_date",
            "imdb_url"] + genre_cols
    movies = pd.read_csv(movies_file, sep="|", names=cols,
                         encoding="latin-1", usecols=range(len(cols)))

    # Build genres string from binary columns
    movies["genres"] = movies[genre_cols].apply(
        lambda row: "|".join([g for g, v in zip(genre_cols, row) if v == 1]),
        axis=1
    )
    movies["year"] = movies["release_date"].str.extract(r"(\d{4})").fillna("N/A")
    return movies[["movie_id", "title", "genres", "year", "imdb_url"]]


def load_ratings():
    """Load and return ratings DataFrame."""
    path = download_movielens()
    ratings_file = os.path.join(path, "u.data")
    ratings = pd.read_csv(
        ratings_file, sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    return ratings


def load_all():
    """Load movies and ratings, return merged DataFrame and raw frames."""
    movies = load_movies()
    ratings = load_ratings()
    return movies, ratings


def get_genre_list(movies: pd.DataFrame) -> list:
    """Extract sorted list of unique genres from movies DataFrame."""
    all_genres = set()
    for genres in movies["genres"]:
        for g in genres.split("|"):
            if g:
                all_genres.add(g)
    return sorted(all_genres)
