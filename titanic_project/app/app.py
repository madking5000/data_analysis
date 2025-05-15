from flask import Flask, request, jsonify
import joblib
import numpy as np
import os


#initialise the flask app
app = Flask(__name__)

# Path: go one level up to models/
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
model = joblib.load(model_path)

#define the prediction end point

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1,-1)
    prediction = model.predict(features)[0]
    return jsonify({'survived': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)