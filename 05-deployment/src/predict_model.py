from flask import Flask
from flask import request

import pickle

def load(filename):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

dv = load('../models/dv.bin')
model = load('../models/model1.bin')

app = Flask('project')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return result


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5050)