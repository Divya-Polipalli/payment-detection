from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final = np.array([features])

    prediction = model.predict(final)

    if prediction[0] == 1:
        result = "🚨 Alert! Fraudulent Transaction Detected"
    else:
        result = "🎉 Woohoo! This Transaction Looks Safe"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)