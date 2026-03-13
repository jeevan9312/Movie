"""
app.py — Main Streamlit application for Movie Recommender
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #e50914, #ff6b35);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 2rem; }
    .movie-card {
        background: #1a1a2e; border: 1px solid #2a2a4a;
        border-radius: 12px; padding: 1rem 1.2rem;
        margin-bottom: 0.8rem; transition: border-color 0.2s;
    }
    .movie-card:hover { border-color: #e50914; }
    .movie-title { font-size: 1.05rem; font-weight: 700; color: #fff; }
    .movie-meta  { font-size: 0.8rem; color: #aaa; margin-top: 0.2rem; }
    .score-badge {
        background: #e50914; color: white;
        padding: 2px 8px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
    }
    .genre-tag {
        background: #2a2a4a; color: #bbb;
        padding: 2px 7px; border-radius: 10px;
        font-size: 0.72rem; margin-right: 4px;
    }
    .stButton > button {
        background: #e50914; color: white;
        border: none; border-radius: 8px;
        font-weight: 600; width: 100%;
    }
    .stButton > button:hover { background: #c40812; }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        border-left: 4px solid #e50914;
        padding-left: 0.6rem; margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model & data (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recommendation engine...")
def load_model():
    from models.recommender import HybridRecommender, MODEL_PATH
    if HybridRecommender.exists():
        return HybridRecommender.load()
    return None

@st.cache_data(show_spinner=False)
def load_data():
    from utils.data_loader import load_movies, get_genre_list
    movies = load_movies()
    genres = get_genre_list(movies)
    return movies, genres


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    user_id = st.text_input("👤 Your Username", value="guest",
                            help="Used to save your favorites to Firebase")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    alpha = st.slider("Algorithm blend", 0.0, 1.0, 0.5,
                      help="0 = full Content-Based | 1 = full Collaborative")

    st.markdown("---")
    st.markdown("**Algorithm**: Hybrid (TF-IDF + SVD)")
    st.markdown("**Dataset**: MovieLens 100K")
    st.markdown("**Database**: Firebase Firestore")
    st.markdown("---")

    page = st.radio("📌 Navigate", ["🔍 Recommendations", "❤️ My Favorites"])


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid ML-powered movie recommendations</div>',
            unsafe_allow_html=True)


def render_movie_card(row, user_id, show_save=True):
    """Render a movie card with save-to-favorites button."""
    with st.container():
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{row['title']}</div>
            <div class="movie-meta">
                📅 {row.get('year','N/A')} &nbsp;|&nbsp;
                {''.join(f'<span class="genre-tag">{g}</span>'
                         for g in str(row.get('genres','')).split('|') if g)}
            </div>
            {'<span class="score-badge">Score: ' + f"{row['score']:.2f}" + '</span>'
             if 'score' in row else ''}
        </div>
        """, unsafe_allow_html=True)

    if show_save:
        if st.button(f"❤️ Save", key=f"save_{row['movie_id']}_{user_id}"):
            from utils.firebase_helper import save_favorite
            if save_favorite(user_id, row.to_dict()):
                st.success("Saved to favorites!")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Recommendations
# ─────────────────────────────────────────────────────────────────────────────
if "Recommendations" in page:
    movies_df, genres = load_data()
    model = load_model()

    if model is None:
        st.error("⚠️ Model not found. Please run `python train_model.py` first.")
        st.code("python train_model.py", language="bash")
        st.stop()

    col1, col2 = st.columns([3, 1])

    with col1:
        all_titles = sorted(movies_df["title"].tolist())
        selected_movie = st.selectbox(
            "🎥 Search for a movie",
            options=all_titles,
            index=all_titles.index("Toy Story (1995)") if "Toy Story (1995)" in all_titles else 0,
            help="Start typing to search"
        )

    with col2:
        genre_filter = st.selectbox("🎭 Filter by genre", ["All"] + genres)

    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Finding your perfect movies..."):
            results = model.hybrid(selected_movie, top_n=top_n, alpha=alpha)

            if genre_filter != "All":
                results = model.filter_by_genre(results, genre_filter)

        if results.empty:
            st.warning("No recommendations found. Try a different movie or genre filter.")
        else:
            st.markdown(f'<div class="section-header">Recommendations for "{selected_movie}"</div>',
                        unsafe_allow_html=True)
            st.caption(f"Showing {len(results)} results"
                       + (f" filtered by **{genre_filter}**" if genre_filter != "All" else ""))

            for _, row in results.iterrows():
                render_movie_card(row, user_id, show_save=True)

    # Also show a quick content-only vs collaborative breakdown
    with st.expander("🔬 Algorithm breakdown (advanced)"):
        if selected_movie:
            cb_res = model.content_based(selected_movie, top_n=5)
            cf_res = model.collaborative(selected_movie, top_n=5)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Content-Based (TF-IDF)**")
                if not cb_res.empty:
                    st.dataframe(cb_res[["title", "score"]].round(3), hide_index=True)
            with c2:
                st.markdown("**Collaborative (SVD)**")
                if not cf_res.empty:
                    st.dataframe(cf_res[["title", "score"]].round(3), hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: My Favorites
# ─────────────────────────────────────────────────────────────────────────────
elif "Favorites" in page:
    st.markdown(f'<div class="section-header">❤️ Favorites for "{user_id}"</div>',
                unsafe_allow_html=True)

    from utils.firebase_helper import get_favorites, remove_favorite
    favorites = get_favorites(user_id)

    if not favorites:
        st.info("No favorites saved yet. Go to Recommendations and save some movies!")
    else:
        st.caption(f"{len(favorites)} movie(s) saved")
        for fav in favorites:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{fav['title']}</div>
                    <div class="movie-meta">
                        📅 {fav.get('year','N/A')} &nbsp;|&nbsp;
                        {''.join(f'<span class="genre-tag">{g}</span>'
                                 for g in str(fav.get('genres','')).split('|') if g)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{fav['movie_id']}"):
                    remove_favorite(user_id, fav["movie_id"])
                    st.rerun()
