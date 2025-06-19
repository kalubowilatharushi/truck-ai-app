from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load inbuilt truck dataset
data = pd.read_csv("inbuilt_truck_data.csv")
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier().fit(X, y)
labels = {0: '🟢 Low', 1: '🟡 Medium', 2: '🔴 High'}

@app.route("/predict", methods=["GET"])
def predict():
    preds = model.predict(X)
    results = []
    for i, row in data.iterrows():
        results.append({
            "id": f"TRK-{i+1:03}",
            "temp": row['Engine_Temp'],
            "pressure": row['Oil_Pressure'],
            "rpm": row['RPM'],
            "mileage": row['Mileage'],
            "risk": labels[preds[i]]
        })
    return jsonify(results)

@app.route("/analyze", methods=["POST"])
def analyze():
    content = request.json.get("text", "")
    if not content.strip():
        return jsonify({"error": "No text provided"}), 400

    vectorizer = TfidfVectorizer(stop_words="english")
    X_text = vectorizer.fit_transform([content])
    scores = X_text.toarray().flatten()
    features = vectorizer.get_feature_names_out()
    keywords = sorted(zip(features, scores), key=lambda x: -x[1])[:5]

    return jsonify({
        "keywords": [kw[0] for kw in keywords],
        "suggestion": "Check coolant system and throttle body."
    })

if __name__ == "__main__":
    app.run(debug=True)
