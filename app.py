
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = pickle.load(open('model_RandomForest.h5', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    #For rendering results on HTML GUI

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Result of prediction is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)

    # apply scaling on input data
    scale = MinMaxScaler()
    scaled_features = scale.fit_transform(data)


    prediction = model.predict([np.array(list(scaled_features.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)