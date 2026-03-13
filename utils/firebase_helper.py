"""
Firebase utility for saving and retrieving user favorites.
Uses Firebase Admin SDK with Firestore.
"""

import os
import streamlit as st

_firebase_initialized = False


def _init_firebase():
    """Initialize Firebase app (only once)."""
    global _firebase_initialized
    if _firebase_initialized:
        return True
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        # Support both local .env and Streamlit Cloud secrets
        if "firebase" in st.secrets:
            import json, tempfile
            key_dict = dict(st.secrets["firebase"])
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(key_dict, f)
                key_path = f.name
        else:
            key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")

        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)

        _firebase_initialized = True
        return True
    except Exception as e:
        st.warning(f"⚠️ Firebase not connected: {e}. Favorites won't be saved.")
        return False


def save_favorite(user_id: str, movie: dict):
    """Save a movie to a user's favorites in Firestore."""
    if not _init_firebase():
        return False
    try:
        from firebase_admin import firestore
        db = firestore.client()
        doc_ref = db.collection("favorites").document(user_id)\
                    .collection("movies").document(str(movie["movie_id"]))
        doc_ref.set({
            "movie_id": movie["movie_id"],
            "title": movie["title"],
            "genres": movie.get("genres", ""),
            "year": movie.get("year", ""),
        })
        return True
    except Exception as e:
        st.error(f"Error saving favorite: {e}")
        return False


def get_favorites(user_id: str) -> list:
    """Retrieve a user's favorite movies from Firestore."""
    if not _init_firebase():
        return []
    try:
        from firebase_admin import firestore
        db = firestore.client()
        docs = db.collection("favorites").document(user_id)\
                  .collection("movies").stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        st.error(f"Error loading favorites: {e}")
        return []


def remove_favorite(user_id: str, movie_id: int):
    """Remove a movie from a user's favorites."""
    if not _init_firebase():
        return False
    try:
        from firebase_admin import firestore
        db = firestore.client()
        db.collection("favorites").document(user_id)\
          .collection("movies").document(str(movie_id)).delete()
        return True
    except Exception as e:
        st.error(f"Error removing favorite: {e}")
        return False
