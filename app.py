from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    age = int(request.form["age"])

    features = np.array([[area, bedrooms, bathrooms, age]])
    prediction = model.predict(features)

    return render_template("index.html", prediction_text=f"Predicted Price: ₹ {prediction[0]:,.2f}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)