"""
Flask Web Application - Movie Recommendation System
Based on Albini (2021) Flask + SVD tutorial structure.
Student modifications: mood filter UI, hybrid scoring, rating submission, explanation panel.
"""

from flask import Flask, render_template, request, jsonify, session
from myrecommender import MovieRecommender
import os, json

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Singleton recommender loaded once, reused across requests
myrecommender = MovieRecommender(n_factors=50)
myrecommender.load_data().fit()
print(f"Model fitted. Users: {myrecommender.get_user_count()}, Movies: {len(myrecommender.get_all_movies())}")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route('/')
def index():
    movies = myrecommender.get_all_movies()
    moods = myrecommender.get_available_moods()
    user_id = session.get('user_id', 1)
    return render_template('index.html', movies=movies, moods=moods, user_id=user_id)


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    user_id = int(data.get('user_id', 1))
    mood = data.get('mood', None)
    n = int(data.get('n', 10))
    alpha = float(data.get('alpha', 0.7))

    session['user_id'] = user_id

    recs = myrecommender.recommend_for_user(user_id=user_id, n=n, mood=mood, alpha=alpha)
    return jsonify({'recommendations': recs, 'user_id': user_id, 'mood': mood})


@app.route('/api/similar/<int:movie_id>')
def get_similar(movie_id):
    similar = myrecommender.similar_movies(movie_id=movie_id, n=8)
    movie_info = myrecommender.movies_df[myrecommender.movies_df['movie_id'] == movie_id]
    title = movie_info.iloc[0]['title'] if not movie_info.empty else 'Unknown'
    return jsonify({'movie_id': movie_id, 'title': title, 'similar': similar})


@app.route('/api/rate', methods=['POST'])
def submit_rating():
    data = request.get_json()
    user_id = int(data.get('user_id', 1))
    movie_id = int(data.get('movie_id'))
    rating = float(data.get('rating'))

    if not (1.0 <= rating <= 5.0):
        return jsonify({'error': 'Rating must be between 1 and 5'}), 400

    myrecommender.add_rating(user_id=user_id, movie_id=movie_id, rating=rating)
    return jsonify({'status': 'ok', 'message': f'Rated movie {movie_id} with {rating}/5.0'})


@app.route('/api/movies')
def list_movies():
    movies = myrecommender.get_all_movies()
    return jsonify({'movies': movies})


@app.route('/api/stats/<int:movie_id>')
def movie_stats(movie_id):
    stats = myrecommender.get_movie_stats(movie_id)
    return jsonify(stats)


@app.route('/api/moods')
def list_moods():
    return jsonify({'moods': myrecommender.get_available_moods()})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
