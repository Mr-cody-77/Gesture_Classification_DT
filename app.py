from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("joystick_gesture_model.pkl")

@app.route("/", methods=["GET"])
def index():
    return "Gesture ML API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    vx = data.get("Vx")
    vy = data.get("Vy")
    prediction = model.predict([[vx, vy]])
    return jsonify({"gesture": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
