import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# Starting the app
app = Flask(__name__)

# Load the model
model = pickle.load(open("best_model.pkl", "rb"))

# Load the scaler
scaler = pickle.load(open("boston_scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(scaled_data)
    print(output)
    return jsonify(output[0])

@app.route("/predict", methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT = data
    LSTAT = np.log(LSTAT)
    CRIM = np.log(CRIM)
    DIS = np.log(DIS)
    RM_squared = RM**2
    LSTAT_squared = LSTAT**2
    RM_LSTAT = RM*LSTAT
    NOX_DIS = NOX*DIS
    INDUS_NOX = INDUS*NOX
    TAX_RAD = TAX*RAD
    ROOMS_PER_DWELLING = (RM / AGE)
    RAD_binary = int(RAD >= 24)
    new_data = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, RM_squared, LSTAT_squared, RM_LSTAT, NOX_DIS, INDUS_NOX, TAX_RAD, ROOMS_PER_DWELLING, RAD_binary]
    scaled_input = scaler.transform(np.array(list(new_data)).reshape(1, -1))
    print(scaled_input)
    output = model.predict(scaled_input)[0]
    return render_template("home.html", prediction_text="The predicted House price is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)