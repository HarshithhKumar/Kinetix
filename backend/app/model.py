import tensorflow as tf

from .config import MODEL_PATH

model = tf.keras.models.load_model(str(MODEL_PATH))

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def predict_activity(data):
    prediction = model.predict(data, verbose=0)

    predicted_class = int(prediction.argmax(axis=1)[0])
    confidence = float(prediction[0][predicted_class])
    activity = ACTIVITY_NAMES[predicted_class]

    probabilities = {
        ACTIVITY_NAMES[i]: float(prediction[0][i])
        for i in range(len(ACTIVITY_NAMES))
    }

    return activity, confidence, probabilities
