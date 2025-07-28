from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model safely
try:
    model = joblib.load("gesture_model.pkl")
    print("✅ Model loaded successfully")
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

        # Extract all features (support for more than just Vx, Vy)
        features = data.get("features")

        if features is None or not isinstance(features, list):
            return jsonify({"error": "Missing or invalid 'features' field"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        prediction = model.predict([features])
        return jsonify({"gesture": prediction[0]})

    except Exception as e:
        print("❌ Error in prediction:", e)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
