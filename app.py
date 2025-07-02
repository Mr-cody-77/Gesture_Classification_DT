from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model safely
try:
    model = joblib.load("joystick_gesture_model.pkl")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

@app.route("/", methods=["GET"])
def index():
    return "Gesture ML API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        vx = data.get("Vx")
        vy = data.get("Vy")

        if vx is None or vy is None:
            return jsonify({"error": "Missing Vx or Vy"}), 400

        # Ensure model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        prediction = model.predict([[vx, vy]])
        return jsonify({"gesture": prediction[0]})

    except Exception as e:
        print("❌ Error in prediction:", e)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
