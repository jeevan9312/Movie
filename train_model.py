import sys
import os
 
# Fix for Windows: ensure absolute project root is in sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
 
from utils.data_loader import load_all
from models.recommender import HybridRecommender
 
 
def main():
    print("=" * 50)
    print("  Movie Recommender — Model Training")
    print("=" * 50)
 
    # 1. Load data
    print("\n[1/3] Loading MovieLens 100K dataset...")
    movies, ratings = load_all()
    print(f"      ✓ {len(movies)} movies loaded")
    print(f"      ✓ {len(ratings)} ratings loaded")
    print(f"      ✓ {ratings['user_id'].nunique()} unique users")
 
    # 2. Train model
    print("\n[2/3] Training hybrid model (Content + Collaborative)...")
    model = HybridRecommender()
    model.fit(movies, ratings)
    print("      ✓ Content-based (TF-IDF) trained")
    print("      ✓ Collaborative (SVD, 50 components) trained")
 
    # 3. Save model
    print("\n[3/3] Saving model...")
    model.save()
    print("      ✓ Model saved to models/hybrid_model.pkl")
 
    # 4. Quick sanity test
    print("\n--- Sanity Check ---")
    test_movie = "Toy Story (1995)"
    recs = model.hybrid(test_movie, top_n=5)
    print(f"Top 5 recommendations for '{test_movie}':")
    for _, row in recs.iterrows():
        print(f"  • {row['title']} ({row['year']}) — score: {row['score']:.3f}")
 
    print("\n✅ Training complete! Run: streamlit run app.py")
 
 
if __name__ == "__main__":
    main()