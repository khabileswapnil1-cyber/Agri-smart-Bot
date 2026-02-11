import os
import joblib
import numpy as np
import google.generativeai as genai
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# तुमची Google Gemini API Key येथे टाका
genai.configure(api_key="AIzaSyDKuwoOX3DthNEmoO7dpVUxQN_CuVAK0yg")
ai_model = genai.GenerativeModel('gemini-2.5-flash')

# अधिकृत सरकारी स्रोत
OFFICIAL_SOURCES = {
    "weather": "https://mausam.imd.gov.in/",
    "market": "https://agmarknet.gov.in/",
    "soil": "https://soilhealth.dac.gov.in/"
}

# ML मॉडेल लोड करणे
try:
    ml_model = joblib.load('crop_model.pkl')
except:
    ml_model = None
    print("Warning: crop_model.pkl not found. Using simulation mode.")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/analyze", methods=['POST'])
def analyze():
    try:
        data = request.json
        location = data.get('location', 'Maharashtra')
        n = float(data['n'])
        p = float(data['p'])
        k = float(data['k'])
        ph = float(data['ph'])

        # हवामान आणि पावसाचा अंदाज (Average for 2026 Prediction)
        temp, hum, rain = 28.5, 75.0, 1100.0
        features = np.array([[n, p, k, temp, hum, ph, rain]])

        # १. टॉप ५ पिकांची शिफारस (ML Logic)
        if ml_model and hasattr(ml_model, "predict_proba"):
            probs = ml_model.predict_proba(features)[0]
            top_indices = np.argsort(probs)[-5:][::-1]
            top_crops = ml_model.classes_[top_indices]
        else:
            # मॉडेल नसल्यास डिफॉल्ट टॉप पिके
            top_crops = ['Soybean', 'Cotton', 'Rice', 'Pigeonpeas', 'Gram']

        # २. Generative AI कडून सविस्तर मार्केट आणि हवामान विश्लेषण
        prompt = f"""
        User Location: {location}. 
        Top 5 Recommended Crops: {', '.join(top_crops)}.
        Soil: N={n}, P={p}, K={k}, pH={ph}.
        
        Provide a detailed agricultural report in Marathi:
        1. Why these 5 crops are perfect for this soil?
        2. Market price history of these crops for the last 2 years (2024-2026).
        3. Future weather suitability based on official IMD trends.
        4. Clear reasoning for each crop's potential profit.
        Format with bullet points. Keep it professional and encouraging.
        """
        
        response = ai_model.generate_content(prompt)
        ai_advice = response.text

        return jsonify({
            "status": "success",
            "location": location,
            "crops": [c.upper() for c in top_crops],
            "ai_advice": ai_advice,
            "links": OFFICIAL_SOURCES
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)