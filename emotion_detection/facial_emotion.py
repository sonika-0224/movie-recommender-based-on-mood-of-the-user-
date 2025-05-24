from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os

def train_facial_model():
    train_dir = 'data/facial/train'
    test_dir = 'data/facial/test'

    # Image generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical'
    )

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if not os.path.exists("models"):
        os.mkdir("models")

    checkpoint = ModelCheckpoint("models/facial_emotion.h5", save_best_only=True, monitor="val_accuracy", mode="max")

    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=10,
        callbacks=[checkpoint]
    )

    print("✅ Model training complete. Saved to models/facial_emotion.h5")

from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('models/facial_emotion.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_facial_emotion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"❌ Failed to load image at: {image_path}")

    try:
        face = cv2.resize(img, (48, 48))
    except Exception as e:
        raise ValueError(f"❌ Could not resize image: {e}")

    face = face.reshape(1, 48, 48, 1) / 255.0
    pred = model.predict(face)
    return emotion_labels[np.argmax(pred)]


