# CineMatch-Hybrid Movie Recommendation System

**MSAI-631 Assignment | Built on Albini (2021) SVD + Flask Foundation**

---

## Project Overview

CineMatch is a hybrid movie recommendation web application that combines **Singular Value Decomposition (SVD)** collaborative filtering with **content-based genre similarity** scoring. The system is built on the tutorial foundation by Gabriele Albini (2023), with substantial extensions to demonstrate a deeper understanding of recommendation system architecture.

### Foundation (Albini Tutorial, 2023)

The original system sampled the **Netflix** dataset, trained SVD via the SURPRISE library, pre-computed all predictions offline into **MongoDB** and served them through a Flask app that read from MongoDB at runtime.

### How This Project Differs

- Re-implements SVD directly with **`scipy.sparse.linalg.svds`** (no Surprise) so the math is visible
- **Live in-memory inference** at request time (no MongoDB) - new ratings update results instantly
- Adds a **content-based genre signal** and blends it with SVD (hybrid, alpha-weighted)
- CSV-based. no database dependency

### Student Extensions


| Extension                  | Description                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Hybrid Scoring**         | Weighted blend of SVD score (collaborative) and genre cosine similarity (content-based), controlled by a UI slider |
| **Mood-Based Filtering**   | Six mood categories mapped to genre clusters; post-filters SVD results before ranking                              |
| **Confidence Weighting**   | Movies with <30 ratings receive a shrinkage penalty to avoid over-recommending sparse items                        |
| **Cold-Start Handler**     | New/unknown users receive popularity-weighted recommendations using a Bayesian-style score                         |
| **Explanation Engine**     | Every recommendation includes a natural-language reason generated from the scoring components                      |
| **Live Rating Submission** | Users can rate any movie through the UI; model marks itself for refit                                              |
| **Similar Movies Panel**   | Content-based genre similarity shown for any selected movie                                                        |


---

## Architecture

```
CineMatch/
- app.py              # Flask application (routes, API endpoints)
- recommender.py      # Core ML engine (SVD + content-based hybrid)
- requirements.txt    # Python dependencies
- data/
   - movies.csv      # 200 movies with genre labels
   - ratings.csv     # ~20,000 synthetic ratings (500 users)
- templates/
    - index.html      # Single-page web UI (HTML/CSS/JS)

```

### Algorithm Flow

```
User Request
    |
SVD Predicted Ratings ------------------>(Collaborative Filtering) | Hybrid Score = α×SVD + (1-α)×Genre × Confidence
                                        
Genre Cosine Similarity ---------------->(Content-Based) | Mood Filter (optional)---> Top-N Results with Explanations
		

```

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cinematch-recommender.git
cd cinematch-recommender

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser to `http://localhost:5000`

---

## Usage

1. **Enter a User ID** (1-500) in the control panel
2. **Select a mood** (optional) to filter results by genre cluster
3. **Adjust the hybrid slider** - move left for content-based, right for collaborative filtering
4. **Click "Get Picks"** to generate recommendations
5. **Click any movie card** to see similar films and submit a rating

---

## API Endpoints


| Method | Endpoint                  | Description                       |
| ------ | ------------------------- | --------------------------------- |
| POST   | `/api/recommend`          | Get personalized recommendations  |
| GET    | `/api/similar/<movie_id>` | Get content-similar movies        |
| POST   | `/api/rate`               | Submit a user rating              |
| GET    | `/api/movies`             | List all movies                   |
| GET    | `/api/stats/<movie_id>`   | Get rating statistics for a movie |
| GET    | `/api/moods`              | List available mood filters       |


### Example Request

```json
POST /api/recommend
{
  "user_id": 42,
  "mood": "tense",
  "n": 10,
  "alpha": 0.7
}
```

---

## Technical Details

### SVD Decomposition

The rating matrix R (500 users × 200 movies) is decomposed as:

```
R = U × Σ × V^T
```

Where:

- **U** = user latent factors (500 × k)
- **Σ** = singular values (k × k diagonal)
- **V^T** = item latent factors (k × 200)
- **k** = 50 latent dimensions (configurable)

### Hybrid Score Formula

```
hybrid_score = (α × svd_score + (1-α) × genre_similarity × 5.0) × confidence_weight
```

Where `confidence_weight = min(1.0, rating_count / 30.0)`

### Genre Similarity

Computed using cosine similarity on multi-label binary genre vectors (19 genre dimensions via `MultiLabelBinarizer`).

---

## References

Albini, G. (2023). *Building a movie recommender web app from scratch with SVD and Flask* (Parts 1 & 2) [Blog post]. Medium. [https://gabri-albini.medium.com/](https://gabri-albini.medium.com/) Source code: [https://github.com/gabri-al/recommender_system](https://github.com/gabri-al/recommender_system)

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, *42*(8), 30–37. [https://doi.org/10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)

Ricci, F., Rokach, L., & Shapira, B. (2022). *Recommender systems: Techniques, evaluation, and challenges*. Springer.

Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems*, *5*(4), 1–19. [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)

---



