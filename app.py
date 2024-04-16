import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    features = [x for x in request.form.values()]
    for i in range(len(features)):
        if i != 3 and i != 4:
            features[i] = int(features[i])

    print((features))

    pred = model.predict([features])

    return render_template("index.html", prediction_text = "Fertilizer:{}".format(pred))


if __name__ == '__main__':
    app.run(debug = True)
