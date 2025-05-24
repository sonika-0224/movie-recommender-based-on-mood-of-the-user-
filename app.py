from flask import Flask, request, render_template
from emotion_detection.text_emotion import predict_emotion
from recommend_movies import recommend_movies
from emotion_detection.audio_emotion import predict_audio_emotion
from emotion_detection.facial_emotion import predict_facial_emotion
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text_input = request.form['user_input']
        mood = predict_emotion(text_input)
        recommendations = recommend_movies(mood)
        return render_template('result.html', mood=mood, movies=recommendations.to_dict(orient='records'))
    return render_template('index.html')

@app.route('/audio', methods=['GET', 'POST'])
def audio_input():
    if request.method == 'POST':
        audio = request.files['audio_file']
        audio_path = os.path.join('uploads/audio', audio.filename)
        audio.save(audio_path)

        try:
            mood = predict_audio_emotion(audio_path)
            recommendations = recommend_movies(mood)
        except Exception as e:
            return f"Error processing audio: {str(e)}"

        return render_template('result.html', mood=mood, movies=recommendations.to_dict(orient='records'))
    return render_template('audio.html')

@app.route('/face', methods=['GET', 'POST'])
def face():
    if request.method == 'POST':
        img_file = request.files['face_image']
        file_path = os.path.join('uploads/faces', img_file.filename)
        img_file.save(file_path)

        mood = predict_facial_emotion(file_path)
        recommendations = recommend_movies(mood)

        return render_template('result.html', mood=mood, movies=recommendations.to_dict(orient='records'))
    return render_template('face.html')


@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


if __name__ == '__main__':
    app.run(debug=True, port=5050)

