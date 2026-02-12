import os
import joblib
import numpy as np
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# It is best practice to use environment variables for keys on Render
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDKuwoOX3DthNEmoO7dpVUxQN_CuVAK0yg")
genai.configure(api_key=API_KEY, transport='rest')
ai_model = genai.GenerativeModel('gemini-pro')

OFFICIAL_SOURCES = {
    "weather": "https://mausam.imd.gov.in/",
    "market": "https://agmarknet.gov.in/",
    "soil": "https://soilhealth.dac.gov.in/"
}

# Ensure the model is loaded from the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'crop_model.pkl')

try:
    ml_model = joblib.load(model_path)
    print("Crop model loaded successfully.")
except Exception as e:
    ml_model = None
    print(f"Warning: crop_model.pkl not found. Simulation mode active. Error: {e}")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/analyze", methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "माहिती मिळालेली नाही."})

        # Convert inputs safely to float, default to 0 if empty
        location = data.get('location', 'Maharashtra')
        n = float(data.get('n') or 0)
        p = float(data.get('p') or 0)
        k = float(data.get('k') or 0)
        ph = float(data.get('ph') or 7.0)

        # Environmental constants (Can be replaced with real API data later)
        temp, hum, rain = 28.5, 75.0, 1100.0

        # ML Prediction Logic
        column_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=column_names)

        if ml_model and hasattr(ml_model, "predict_proba"):
            probs = ml_model.predict_proba(features)[0]
            top_indices = np.argsort(probs)[-5:][::-1]
            top_crops = ml_model.classes_[top_indices]
        else:
            top_crops = ['Soybean', 'Cotton', 'Rice', 'Pigeonpeas', 'Gram']

        # AI Report Generation
        prompt = f"""
        User Location: {location}.
        Top 5 Recommended Crops: {', '.join(top_crops)}.
        Soil data: N={n}, P={p}, K={k}, pH={ph}.
       
        Provide a detailed agricultural report in Marathi:
        1. Why these 5 crops suit this specific soil?
        2. Market price trends for 2024-2026.
        3. Weather suitability based on IMD trends.
        4. Profit potential for each.
        Use bullet points. Keep it professional.
        """
       
        response = ai_model.generate_content(prompt)
        ai_advice = response.text if response else "AI विश्लेषण उपलब्ध नाही."

        return jsonify({
            "status": "success",
            "location": location,
            "crops": [str(c).upper() for c in top_crops],
            "ai_advice": ai_advice,
            "links": OFFICIAL_SOURCES
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)