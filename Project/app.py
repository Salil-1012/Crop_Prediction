from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Nitrogen = int(request.form['Nitrogen'])
    Phosphrus = int(request.form['Phosphrus'])
    Potassium = int(request.form['Potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    Rainfall = float(request.form['Rainfall'])
    
    input_features = np.array([[Nitrogen, Phosphrus, Potassium, temperature, humidity, ph, Rainfall]])
    
    prediction = model.predict(input_features)
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    
    app.run(debug=True)

