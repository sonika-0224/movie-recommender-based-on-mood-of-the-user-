import pandas as pd

# Map emotions to clusters — customize as needed!
MOOD_CLUSTER_MAP = {
    'happy': [1],
    'sad': [2],
    'angry': [0],
    'neutral': [3],
    'surprised': [4],
    'fear': [5],
    'disgust': [6]
}

def load_movies():
    return pd.read_csv("data/movies/movie_clusters.csv")

def recommend_movies(emotion, top_n=5):
    df = load_movies()
    clusters = MOOD_CLUSTER_MAP.get(emotion.lower(), [3])  # fallback to neutral
    recommended = df[df['cluster'].isin(clusters)]
    
    # Sort by vote average or popularity
    recommended = recommended.sort_values(by='vote_average', ascending=False)
    return recommended[['title', 'vote_average', 'overview']].head(top_n)
