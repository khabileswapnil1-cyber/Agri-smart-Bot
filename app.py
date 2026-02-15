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
client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.0-flash-lite-preview-02-05" # अद्ययावत आणि वेगवान मॉडेल

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
    print(f"Warning: crop_model.pkl not found. AI-only mode active. Error: {e}")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/analyze", methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "माहिती मिळालेली नाही."})

        # युजरकडून मिळालेला डेटा
        location = data.get('location', 'Maharashtra')
        n = float(data.get('n') or 0)
        p = float(data.get('p') or 0)
        k = float(data.get('k') or 0)
        ph = float(data.get('ph') or 7.0)

        # १. ML मॉडेलद्वारे प्राथमिक पिकांचा अंदाज (जर उपलब्ध असेल तर)
        top_crops_list = []
        if ml_model and hasattr(ml_model, "predict_proba"):
            # सरासरी हवामान घटक गृहीत धरले आहेत
            features = pd.DataFrame([[n, p, k, 28.5, 75.0, ph, 1100.0]], 
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            probs = ml_model.predict_proba(features)[0]
            top_indices = np.argsort(probs)[-5:][::-1]
            top_crops_list = [str(c).upper() for c in ml_model.classes_[top_indices]]
        else:
            top_crops_list = ["Soybean", "Cotton", "Tomato", "Pigeonpeas", "Onion"]

        # २. AI अहवाल निर्मिती (५०+ पिके, आधुनिक शेती, आणि बाजारभाव विश्लेषण)
        prompt = f"""
        एक प्रगत कृषी तज्ज्ञ म्हणून, {location} या ठिकाणच्या शेतकऱ्यासाठी खालील माहितीवर आधारित सविस्तर मराठी अहवाल तयार करा:
        
        माहिती:
        - ठिकाण: {location}
        - जमिनीचे घटक: नत्र(N):{n}, स्फुरद(P):{p}, पालाश(K):{k}, सामू(pH):{ph}
        - संभाव्य पिके: {', '.join(top_crops_list)} आणि इतर ५०+ भाजीपाला/नगदी पिके.

        अहवालात खालील ५ मुद्द्यांचा समावेश असावा:
        १. शिफारस केलेली सर्वोत्तम ५ पिके: ही पिके या विशिष्ट जमिनीच्या गुणधर्मांशी (Soil Compatibility) कशी जुळतात?
        २. उत्पन्न वाढीचे मार्ग (Yield Increase): प्रत्येक पिकाचे उत्पन्न दुप्पट करण्यासाठी कोणते विशेष खत व्यवस्थापन किंवा प्रक्रिया करावी?
        ३. आधुनिक शेती पद्धती (Modern Farming): ठिबक सिंचन, मल्चिंग पेपर, किंवा संरक्षित शेती (Polyhouse) यांचा वापर या ५ पिकांसाठी कसा करावा?
        ४. हवामान सुसंगतता (IMD Trends): पुढील ५-६ महिन्यांचा {location} मधील हवामान अंदाज लक्षात घेता ही पिके किती सुरक्षित आहेत?
        ५. आर्थिक विश्लेषण: २०२४ ते २०२६ मधील बाजारभाव कल (Market Price Trends) आणि प्रत्येक पिकाची नफा क्षमता (Profit Potential).

        सूचना: अहवाल बुलेट पॉइंट्समध्ये, व्यावसायिक भाषेत आणि पूर्णपणे मराठीत असावा.
        """
        
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        ai_advice = response.text if response else "AI विश्लेषण उपलब्ध नाही."

        return jsonify({
            "status": "success",
            "location": location,
            "recommended_crops": top_crops_list,
            "ai_advice": ai_advice,
            "links": OFFICIAL_SOURCES
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)