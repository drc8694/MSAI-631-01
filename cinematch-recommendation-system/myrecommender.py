"""
Movie Recommendation Engine
=================================================================
Foundation: The system this project is based on Albini, G. (2023). Building a Movie Recommender Web App from Scratch
  with SVD and Flask (Parts 1 & 2). Medium / Towards Data Science.
  GitHub: https://github.com/gabri-al/recommender_system

  What the original system did:
    - Used a sampled portion of the Netflix Prize dataset (~32,000 users x ~1,300 movies).
    - Trained collaborative filtering with the `surprise` libraries SVD() class using biased=False (Funk SVD, solved via ALS).
    - Tuned the latent-factor count with 5-fold cross-validation on RMSE (~0.83), settling on ~70-80 factors.
    - Pre-computed every user and movie prediction offline and stored the results in a MongoDB database.
    - Served predictions through a Flask app that READ them from MongoDB.

What modifications I made:
    1. Algorithm: replaced the `surprise` SVD()/MongoDB pipeline with a transparent, dependency-light SVD using scipy.sparse.linalg.svds.
    so the linear-algebra steps are visible and modifiable in source.
    2. Live inference: Predictions are computed in-memory at request time (no MongoDB), so new ratings update results immediately.
    3. Hybrid scoring: blends SVD collaborative filtering with a content-based genre cosine-similarity signal (alpha-weighted).
    4. Mood-based filtering layer on top of the ranked results.
    5. Confidence weighting by rating count to damp sparse-item noise.
    6. Cold-start handling: new users get a popularity-weighted fallback.
    7. Explanation generation: each recommendation states why it was chosen, drawn from the scoring components.
=================================================================
"""

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

MOOD_GENRE_MAP = {
    "adventurous":  ["Action", "Adventure", "Sci-Fi", "Fantasy"],
    "thoughtful":   ["Drama", "Biography", "History", "Documentary"],
    "lighthearted": ["Comedy", "Animation", "Family", "Romance"],
    "tense":        ["Thriller", "Mystery", "Crime", "Horror"],
    "romantic":     ["Romance", "Comedy", "Drama"],
    "inspired":     ["Biography", "Sport", "Drama", "Music"],
}


class MovieRecommender:
    """
    Hybrid recommendation system combining:
      1. SVD collaborative filtering (Albini foundation)
      2. Content-based genre similarity (My extension)
      3. Mood-based post-filtering (My extension)
      4. Confidence weighting by rating density (My extension)
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.movies_df = None
        self.ratings_df = None
        self.user_ratings_matrix = None
        self.predicted_ratings = None
        self.genre_similarity_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_ids = []
        self.movie_ids = []
        self.rating_counts = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Loading Data
    # ------------------------------------------------------------------
    def load_data(self):
        self.movies_df = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
        self.ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))

        self.rating_counts = (
            self.ratings_df.groupby('movie_id')['rating']
            .count()
            .to_dict()
        )
        return self

    # ------------------------------------------------------------------
    # Build User-Movie Matrix and Run SVD
    # ------------------------------------------------------------------
    def fit(self):
        """
        Core SVD decomposition - the primary technique from Albini's tutorial.
        We decompose R = U * Σ * V^T and use the reconstructed matrix for
        predicting unobserved ratings.
        """
        if self.ratings_df is None:
            self.load_data()

        # Pivot to user × movie matrix
        pivot = self.ratings_df.pivot_table(
            index='user_id', columns='movie_id', values='rating'
        )
        self.user_ids = list(pivot.index)
        self.movie_ids = list(pivot.columns)
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        # Mean-center ratings per user using ACTUAL rated movies only (not zero-padded mean)
        # This is critical: using the zero-padded matrix mean severely deflates user baselines.
        matrix = pivot.fillna(0).values
        user_ratings_mean = np.array([
            self.ratings_df[self.ratings_df['user_id'] == uid]['rating'].mean()
            for uid in self.user_ids
        ])
        matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

        # SVD — keeping n_factors latent dimensions
        k = min(self.n_factors, min(matrix_demeaned.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(matrix_demeaned), k=k)
        sigma_diag = np.diag(sigma)

        # Reconstruct full rating matrix
        raw_predictions = np.dot(np.dot(U, sigma_diag), Vt) + user_ratings_mean.reshape(-1, 1)

        # Normalize each user row to [1,5] scale for display.
        # Raw SVD magnitudes are reliable for ranking but not absolute magnitude
        # in sparse matrices — per-user normalization is standard practice.
        normed = np.zeros_like(raw_predictions)
        for i in range(raw_predictions.shape[0]):
            row = raw_predictions[i]
            rmin, rmax = row.min(), row.max()
            if rmax > rmin:
                normed[i] = 1.0 + 4.0 * (row - rmin) / (rmax - rmin)
            else:
                normed[i] = np.full_like(row, 3.0)
        self.predicted_ratings = normed
        self.user_ratings_matrix = pivot

        # Build genre similarity for content-based component
        self._build_genre_matrix()
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    #  Content-Based: Genre Similarity
    #  My extension: combines with SVD scores for hybrid ranking
    # ------------------------------------------------------------------
    def _build_genre_matrix(self):
        mlb = MultiLabelBinarizer()
        genre_lists = self.movies_df['genres'].str.split('|').tolist()
        genre_matrix = mlb.fit_transform(genre_lists)
        self.genre_similarity_matrix = cosine_similarity(genre_matrix)
        self.genre_labels = mlb.classes_

    def _genre_similarity_for_movie(self, movie_id: int) -> dict:
        """Return genre similarity scores for all movies relative to a seed movie."""
        if movie_id not in self.movies_df['movie_id'].values:
            return {}
        seed_idx = self.movies_df[self.movies_df['movie_id'] == movie_id].index[0]
        scores = self.genre_similarity_matrix[seed_idx]
        return {
            self.movies_df.iloc[i]['movie_id']: float(scores[i])
            for i in range(len(scores))
        }

    # ------------------------------------------------------------------
    # Confidence Weight - my extension
    # ------------------------------------------------------------------
    def _confidence_weight(self, movie_id: int) -> float:
        """
        Movies with very few ratings have unreliable predicted scores.
        Apply a shrinkage factor so sparse movies are ranked lower unless
        the SVD score is very high.
        """
        count = self.rating_counts.get(movie_id, 0)
        return min(1.0, count / 30.0)  # full confidence at 30+ ratings

    # ------------------------------------------------------------------
    # Primary: Collaborative Filtering Recommendations (SVD)
    # ------------------------------------------------------------------
    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        mood: str = None,
        alpha: float = 0.7,
    ) -> list[dict]:
        """
        Hybrid recommendation for an existing user.

        alpha controls the blend:
          alpha=1.0  → pure SVD collaborative filtering (Albini baseline)
          alpha=0.0  → pure content-based
          alpha=0.7  → 70% SVD + 30% genre similarity (our default)
        """
        if not self._is_fitted:
            self.fit()

        if user_id not in self.user_ids:
            return self._cold_start_recommend(n=n, mood=mood)

        user_idx = self.user_ids.index(user_id)
        predicted_row = self.predicted_ratings[user_idx]

        # Movies this user already rated
        already_rated = set(
            self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id']
        )

        # Seed content-based scores from top-rated movies by this user
        top_rated = (
            self.ratings_df[self.ratings_df['user_id'] == user_id]
            .nlargest(3, 'rating')['movie_id']
            .tolist()
        )
        genre_scores = {}
        for seed_id in top_rated:
            for mid, score in self._genre_similarity_for_movie(seed_id).items():
                genre_scores[mid] = genre_scores.get(mid, 0) + score / len(top_rated)

        # Score each candidate movie
        candidates = []
        for idx, movie_id in self.idx_to_movie_id.items():
            if movie_id in already_rated:
                continue
            if movie_id not in self.movies_df['movie_id'].values:
                continue

            svd_score = float(predicted_row[idx])
            genre_score = genre_scores.get(movie_id, 0.0)
            conf = self._confidence_weight(movie_id)

            hybrid_score = (alpha * svd_score + (1 - alpha) * genre_score * 5.0) * conf

            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            candidates.append({
                'movie_id': int(movie_id),
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'predicted_rating': round(min(5.0, max(1.0, svd_score)), 2),
                'hybrid_score': round(hybrid_score, 4),
                'explanation': self._explain(svd_score, genre_score, conf, top_rated, movie_id),
            })

        # Mood filtering - my extension
        if mood and mood.lower() in MOOD_GENRE_MAP:
            target_genres = set(MOOD_GENRE_MAP[mood.lower()])
            mood_candidates = [
                c for c in candidates
                if target_genres & set(c['genres'].split('|'))
            ]
            if len(mood_candidates) >= n:
                candidates = mood_candidates

        candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return candidates[:n]

    # ------------------------------------------------------------------
    #  Cold-Start Handler - my Extension
    # ------------------------------------------------------------------
    def _cold_start_recommend(self, n: int = 10, mood: str = None) -> list[dict]:
        """
        For users with no rating history, fall back to popularity-weighted
        average ratings. Mood filtering still applies.
        """
        stats = (
            self.ratings_df.groupby('movie_id')['rating']
            .agg(['mean', 'count'])
            .reset_index()
        )
        stats = stats[stats['count'] >= 10]  # require minimum credibility
        stats['popularity_score'] = stats['mean'] * np.log1p(stats['count'])

        merged = stats.merge(self.movies_df, on='movie_id')

        if mood and mood.lower() in MOOD_GENRE_MAP:
            target_genres = set(MOOD_GENRE_MAP[mood.lower()])
            merged = merged[
                merged['genres'].apply(lambda g: bool(target_genres & set(g.split('|'))))
            ]

        merged = merged.sort_values('popularity_score', ascending=False).head(n)

        return [
            {
                'movie_id': int(row['movie_id']),
                'title': row['title'],
                'genres': row['genres'],
                'predicted_rating': round(float(row['mean']), 2),
                'hybrid_score': round(float(row['popularity_score']), 4),
                'explanation': f"Highly rated by {int(row['count'])} viewers with an average of {row['mean']:.1f}/5.0",
            }
            for _, row in merged.iterrows()
        ]

    # ------------------------------------------------------------------ #
    #  Content-Based: Similar Movies                                      #
    # ------------------------------------------------------------------ #
    def similar_movies(self, movie_id: int, n: int = 8) -> list[dict]:
        """
        Find movies with the highest genre similarity to a given title.
        Pure content-based — doesn't require user history.
        """
        if not self._is_fitted:
            self.fit()

        scores = self._genre_similarity_for_movie(movie_id)
        if not scores:
            return []

        results = []
        for mid, sim_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if mid == movie_id:
                continue
            movie_row = self.movies_df[self.movies_df['movie_id'] == mid]
            if movie_row.empty:
                continue
            avg_rating = self.ratings_df[self.ratings_df['movie_id'] == mid]['rating'].mean()
            results.append({
                'movie_id': int(mid),
                'title': movie_row.iloc[0]['title'],
                'genres': movie_row.iloc[0]['genres'],
                'similarity_score': round(sim_score, 3),
                'avg_rating': round(float(avg_rating) if not np.isnan(avg_rating) else 0, 2),
            })
            if len(results) >= n:
                break
        return results

    # ------------------------------------------------------------------
    # Explanation Generator - my extension
    # ------------------------------------------------------------------
    def _explain(self, svd_score, genre_score, conf, top_rated_ids, movie_id) -> str:
        seed_titles = []
        for mid in top_rated_ids[:2]:
            row = self.movies_df[self.movies_df['movie_id'] == mid]
            if not row.empty:
                seed_titles.append(row.iloc[0]['title'])

        if svd_score >= 4.0 and genre_score >= 0.5:
            base = "Users with similar tastes loved this, and it matches your genre preferences"
        elif svd_score >= 4.0:
            base = "Collaborative filtering strongly predicts you will enjoy this"
        elif genre_score >= 0.6:
            base = "Very similar in genre to films you have rated highly"
        elif conf < 0.5:
            base = "Limited ratings data, but the genre profile aligns with your history"
        else:
            base = "Recommended based on your viewing pattern"

        if seed_titles:
            base += f" (similar to: {', '.join(seed_titles)})"
        return base

    # ------------------------------------------------------------------
    # Public Utility Methods
    # ------------------------------------------------------------------
    def get_all_movies(self) -> list[dict]:
        if self.movies_df is None:
            self.load_data()
        return self.movies_df.to_dict(orient='records')

    def get_movie_stats(self, movie_id: int) -> dict:
        if self.ratings_df is None:
            self.load_data()
        subset = self.ratings_df[self.ratings_df['movie_id'] == movie_id]['rating']
        return {
            'count': int(len(subset)),
            'mean': round(float(subset.mean()), 2) if len(subset) else 0,
            'std': round(float(subset.std()), 2) if len(subset) > 1 else 0,
        }

    def get_user_count(self) -> int:
        if self.ratings_df is None:
            self.load_data()
        return int(self.ratings_df['user_id'].nunique())

    def get_available_moods(self) -> list[str]:
        return list(MOOD_GENRE_MAP.keys())

    def add_rating(self, user_id: int, movie_id: int, rating: float):
        """Accept a new rating and mark the model as needing refit."""
        if self.ratings_df is None:
            self.load_data()
        new_row = pd.DataFrame([{'user_id': user_id, 'movie_id': movie_id, 'rating': rating}])
        self.ratings_df = pd.concat([self.ratings_df, new_row], ignore_index=True)
        self._is_fitted = False  # force refit on next recommendation call
        # Persist
        self.ratings_df.to_csv(os.path.join(DATA_DIR, 'ratings.csv'), index=False)
