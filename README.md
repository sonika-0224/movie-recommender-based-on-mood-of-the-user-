# 🎬 Mood-Based Movie Recommender

An intelligent, emotion-driven movie recommendation system built with Flask. This application suggests movies based on the user's detected **mood**, captured from:

- 📝 Text input
- 🎤 Audio file (.wav)
- 📷 Facial expression (via image upload or real-time webcam)

The system uses machine learning models for mood classification and clusters movie data to recommend content that resonates emotionally.

---

## 🚀 Features

- **Real-Time Emotion Detection**  
  Analyze user emotion through natural language, audio tones, or facial expressions.
  
- **Movie Clustering & Matching**  
  Movies are clustered by theme and tone, and matched to detected mood clusters.

- **Responsive Web Interface**  
  Styled with Tailwind CSS for a clean, mobile-first layout.

- **Webcam Integration**  
  Capture facial expressions directly from your browser for instant analysis.

- **Hybrid Input Options**  
  Use text, upload audio/images, or switch to live webcam—all in one interface.

---

## 🛠 Tech Stack

| Category        | Tools Used                             |
|----------------|-----------------------------------------|
| **Backend**     | Flask, Python                           |
| **Frontend**    | HTML, Tailwind CSS, JavaScript (vanilla)|
| **ML Models**   | TensorFlow, scikit-learn, Keras         |
| **Audio**       | librosa, soundfile                      |
| **Image/Video** | OpenCV                                  |
| **Data**        | TMDb 5000 Movie Dataset (Kaggle)        |

---


---

## 🧪 Local Setup Instructions

### 1. Clone the Repository

bash:
git clone https://github.com/your-username/mood-movie-recommender.git
cd mood-movie-recommender

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python3 app.py

and then Visit: http://127.0.0.1:5000


---


---
##🌐 Future Enhancements
1. 🔄 Real-time emotion refresh loop

2. 📊 Confidence scores + mood visualization

3. ☁️ Live deployment on Render or Railway

4. 👥 User login & mood history tracking
