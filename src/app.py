from exception import InvalidUsage
from validations import validate_fields
from joblib import load
from flask import Flask, request, jsonify
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer
import numpy as np

#Imported objects from testing env
scaler = None
model = None

def load_objects():
    global model, scaler
    # model variable refers to the global variable
    with open('models/model.joblib', 'rb') as f:
        model = load(f)

    # scaler variable refers to the global variable
    with open('models/scaler.joblib', 'rb') as f:
        scaler = load(f)

app = Flask(__name__)

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    """
    Handles any error raised by the other methods
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/')
def home_endpoint():
    return "Welcome to the ECOVALUE's clustering model API!"


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        entry = request.get_json()  # Get data posted as a json
        if validate_fields(entry):
            values = list(entry.values())
            datapoint = np.array(values)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
            scaler.transform(datapoint) # scaling data to fit the training limits 
            prediction = model.predict(datapoint)  # runs globally loaded model on the data
            response = jsonify({"cluster": int(prediction[0])})
            return response
        raise InvalidUsage("Unprocessable entity", 422)        
    raise InvalidUsage("Method Not Allowed", 405)


if __name__ == '__main__':
    load_objects()  # load model at the beginning once only
    #app.run(host='0.0.0.0', port=80)
    app.run(host='127.0.0.1', port=8080)
