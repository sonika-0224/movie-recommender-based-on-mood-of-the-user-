import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import os

def preprocess_and_cluster_movies():
    df = pd.read_csv("data/movies/tmdb_5000_movies.csv")
    
    # Combine genres and overview to get mood-relevant features
    df['combined_features'] = df['genres'] + ' ' + df['overview']
    
    # Clean text
    df['combined_features'] = df['combined_features'].fillna('').str.lower()
    
    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

    # Cluster using KMeans
    kmeans = KMeans(n_clusters=7, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    # Save clustered movie data
    if not os.path.exists('models'):
        os.mkdir('models')
    joblib.dump(kmeans, 'models/movie_kmeans.pkl')
    df.to_csv("data/movies/movie_clusters.csv", index=False)
    print("✅ Movies clustered and saved to data/movies/movie_clusters.csv")

if __name__ == "__main__":
    preprocess_and_cluster_movies()
