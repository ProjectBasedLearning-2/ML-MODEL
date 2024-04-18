# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import StandardScaler
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)
# model = pickle.load(open("model.pkl", "rb"))
# print(model)

# @app.route("/predict", methods = ["POST"])
# def predict():
#     # temperature = request.json["Temperature"]
#     # humidity = request.json["Humidity"]
#     # moisture = request.json["Moisture"]
#     # soilType = request.json["SoilType"]
#     # cropType = request.json["CropType"]
#     # n = request.json["N"]
#     # p = request.json["p"]
#     # k = request.json["K"]
#     # features = [(int)(temperature), (int)(humidity), (int)(moisture), soilType, cropType, (int)(n), (int)(p), (int)(k)]
#     features = [x for x in request.form.values()]
#     for i in range(len(features)):
#         if i != 3 and i != 4:
#             features[i] = int(features[i])
#     print(features)
#     pred = model.predict([features])
#     prediction_text = {
#         "prediction": pred
#     }
#     return jsonify(prediction_text)

# if __name__ == '__main__':
#     app.run(debug = True, port = 5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# Load the model pipeline from the pickle file
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    print(data)
    # Ensure that the JSON data contains all the required fields
    required_fields = ['Temperature', 'Humidity', 'Moisture', 'SoilType', 'CropType', 'N', 'P', 'K']
    if not all(key in data for key in required_fields):
        missing_fields = [field for field in required_fields if field not in data]
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400
        
    # Prepare the input features for prediction
    try:
        
        features = [
        data['Temperature'],
        data['Humidity'],
        data['Moisture'],
        data['SoilType'],
        data['CropType'],
        data['N'],
        data['P'],
        data['K']
        ]
        
        

        # Make predictions using the loaded pipeline
        prediction = pipeline.predict([features])
        prediction_list = prediction.tolist()
        prediction_text = {"message": prediction_list}
        return jsonify(prediction_text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

