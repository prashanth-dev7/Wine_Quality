import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('Wine_Quality_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Predicting and returning the salary."""
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)

    if prediction[0] == 0:
        output="Poor Quality Wine"
    else:
        output="Rich Quality Wine"

    return render_template('Index.html', prediction_text="Parameters: {} --> {}".format(final_features,output))

if __name__=="__main__":
    app.run(debug=True)