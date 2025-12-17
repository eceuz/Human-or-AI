from flask import Flask, render_template, request
import joblib
import re
import numpy as np
import os

app = Flask(__name__)

try:
    vectorizer = joblib.load('model_storage/tfidf_vectorizer.pkl')
    models = {
        "Naive Bayes": joblib.load('model_storage/NaiveBayes_model.pkl'),
        "Random Forest": joblib.load('model_storage/RandomForest_model.pkl'),
        "Linear SVC": joblib.load('model_storage/LinearSVC_model.pkl'),
        "Logistic Regression": joblib.load('model_storage/LogisticRegression_model.pkl')
    }
except FileNotFoundError:
    print("Model bulunamadi.")
    models = None

def clean_input_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s%]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_prediction(model, text_vector):
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_vector)[0]
        prob_ai = proba[1] * 100
        prob_human = proba[0] * 100
    else: 
        score = model.decision_function(text_vector)[0]
        prob_ai = (1 / (1 + np.exp(-score))) * 100
        prob_human = 100 - prob_ai
    return prob_ai, prob_human

@app.route('/', methods=['GET', 'POST'])
def home():
    result_data = []
    input_text = ""
    final_verdict = ""

    if request.method == 'POST':
        input_text = request.form.get('text_input', '')
        
        if input_text.strip() and models:
            cleaned_text = clean_input_text(input_text)
            print(f"--- DEBUG: Temiz Metin Uzunluğu: {len(cleaned_text)}")
            print(f"--- DEBUG: Temiz Metin Başlangıcı: {cleaned_text[:50]}")
            input_vector = vectorizer.transform([cleaned_text])
            
            ai_votes = 0
            for name, model in models.items():
                ai_prob, human_prob = get_prediction(model, input_vector)
                prediction = "AI" if ai_prob > human_prob else "HUMAN"
                if prediction == "AI": ai_votes += 1
                
                result_data.append({
                    "model": name, "prediction": prediction,
                    "ai_prob": round(ai_prob, 2), "human_prob": round(human_prob, 2)
                })
            
            final_verdict = "AI" if ai_votes >= 2 else "HUMAN"

    return render_template('index.html', results=result_data, original_text=input_text, verdict=final_verdict)

if __name__ == '__main__':
    app.run(debug=True)