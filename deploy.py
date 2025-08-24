from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Feature names and placeholder values
features_info = [
    ('residual sugar', 5.0),
    ('chlorides', 0.045),
    ('total sulfur dioxide', 120),
    ('density', 0.995),
    ('pH', 3.2),
    ('alcohol', 10.5)
]

@app.route('/')
def home():
    return render_template('index.html', features_info=features_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features in the correct order
        features = [float(request.form[name]) for name, _ in features_info]
        features = np.array([features])  # shape (1,6)
        
        # RandomForest does not need scaling
        # features = scaler.transform(features)  # Not needed

        # Make prediction
        prediction = model.predict(features)[0]
        result = "Good" if prediction == 1 else "Bad"

        return render_template('index.html', features_info=features_info, prediction=result)

    except Exception as e:
        return render_template('index.html', features_info=features_info, prediction=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
