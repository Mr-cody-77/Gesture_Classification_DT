from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
try:
    model = joblib.load("gesture_raw_model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

@app.route("/", methods=["GET"])
def index():
    return "Gesture ML API is live using raw Ax, Ay, Az data!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        raw_data = data.get("raw_data")  # expecting list of [Ax, Ay, Az]

        if raw_data is None or not isinstance(raw_data, list) or not all(len(row) == 3 for row in raw_data):
            return jsonify({"error": "Invalid input format. Expecting 'raw_data': [[Ax, Ay, Az], ...]"}), 400

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Flatten 2D list [ [Ax,Ay,Az], [Ax,Ay,Az], ... ] → [Ax1,Ay1,Az1,Ax2,Ay2,Az2,...]
        flat_features = np.array(raw_data).flatten()

        prediction = model.predict([flat_features])
        return jsonify({"gesture": prediction[0]})

    except Exception as e:
        print("❌ Error in prediction:", e)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
