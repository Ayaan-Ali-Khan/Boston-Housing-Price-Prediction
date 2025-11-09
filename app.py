import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# Starting the app
app = Flask(__name__)

# Load the model
model = pickle.load(open("boston_reg.pkl", "rb"))

# Load the scaler
scaler = pickle.load(open("boston_scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(scaled_data)
    print(output)
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)