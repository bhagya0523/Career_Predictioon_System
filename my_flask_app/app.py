from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/partners')
def partners():
    return render_template('partners.html')

@app.route('/taketest')
def taketest():
    return render_template('taketest.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
