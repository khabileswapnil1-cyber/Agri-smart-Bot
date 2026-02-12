import os
import joblib
import numpy as np
import pandas as pd
from google import genai
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# API Key सुरक्षित ठेवण्यासाठी environment variable वापरा
API_KEY = os.environ.get("GEMINI_API_KEY")
# नवीन पद्धत: क्लायंट तयार करा
client = genai.Client(api_key=API_KEY)
# मॉडेलचे नाव एका व्हेरिएबलमध्ये ठेवा
MODEL_ID = "gemini-2.0-flash"

OFFICIAL_SOURCES = {
    "weather": "https://mausam.imd.gov.in/",
    "market": "https://agmarknet.gov.in/",
    "soil": "https://soilhealth.dac.gov.in/"
}

# मॉडेल लोड करणे
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

        location = data.get('location', 'Maharashtra')
        n = float(data.get('n') or 0)
        p = float(data.get('p') or 0)
        k = float(data.get('k') or 0)
        ph = float(data.get('ph') or 7.0)

        temp, hum, rain = 28.5, 75.0, 1100.0

        column_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=column_names)

        if ml_model and hasattr(ml_model, "predict_proba"):
            probs = ml_model.predict_proba(features)[0]
            top_indices = np.argsort(probs)[-5:][::-1]
            top_crops = ml_model.classes_[top_indices]
        else:
            top_crops = ['Soybean', 'Cotton', 'Rice', 'Pigeonpeas', 'Gram']

        # AI Report Generation Prompt
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
        
        # दुरुस्त केलेला भाग: क्लायंट वापरून content generate करणे
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        
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