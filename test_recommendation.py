from recommend_movies import recommend_movies

emotion = "happy"
movies = recommend_movies(emotion)
print(f"\nTop movies for mood '{emotion}':\n")
print(movies)
