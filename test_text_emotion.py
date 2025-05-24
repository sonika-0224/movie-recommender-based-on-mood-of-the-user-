from emotion_detection.text_emotion import predict_emotion

text = "I'm feeling incredibly joyful today!"
emotion = predict_emotion(text)
print(f"Predicted Emotion: {emotion}")
