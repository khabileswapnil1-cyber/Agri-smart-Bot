import os
import joblib
import numpy as np
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDKuwoOX3DthNEmoO7dpVUxQN_CuVAK0yg")

# Force the library to use the v1beta endpoint where gemini-1.5-flash is most stable
genai.configure(api_key=API_KEY, transport='rest')

# Official sources for the report
OFFICIAL_SOURCES = {
    "weather": "https://mausam.imd.gov.in/",
    "market": "https://agmarknet.gov.in/",
    "soil": "https://soilhealth.dac.gov.in/"
}

# ML Model Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'crop_model.pkl')

try:
    ml_model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    ml_model = None
    print(f"ML Model not found, using simulation mode. Error: {e}")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/analyze", methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        location = data.get('location', 'Maharashtra')
        n = float(data.get('n') or 0)
        p = float(data.get('p') or 0)
        k = float(data.get('k') or 0)
        ph = float(data.get('ph') or 7.0)

        # 1. ML Logic
        column_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features = pd.DataFrame([[n, p, k, 28.0, 70.0, ph, 1000.0]], columns=column_names)

        if ml_model:
            probs = ml_model.predict_proba(features)[0]
            top_indices = np.argsort(probs)[-5:][::-1]
            top_crops = ml_model.classes_[top_indices]
        else:
            top_crops = ['Soybean', 'Cotton', 'Rice', 'Pigeonpeas', 'Gram']

        # 2. AI Logic (Updated to fix 404 error)
        try:
            # Using the direct model string
            model = genai.GenerativeModel('gemini-1.5-flash')
           
            prompt = f"""
            Location: {location}. Soil: N={n}, P={p}, K={k}, pH={ph}.
            Recommended Crops: {', '.join(top_crops)}.
            Write a professional agricultural report in Marathi including:
            - Suitability reasons
            - Price trends (2024-2026)
            - Weather and Profit tips.
            """
           
            response = model.generate_content(prompt)
            ai_advice = response.text
        except Exception as ai_err:
            print(f"AI Error: {ai_err}")
            ai_advice = "क्षमस्व, AI विश्लेषण सध्या उपलब्ध नाही. कृपया पिकांची यादी पहा."

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