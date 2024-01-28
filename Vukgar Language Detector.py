from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


def go(b):
    a = [b]
    prediction = model.predict(a)
    if prediction == 1:
        return 0
    else:
        return 1


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = go(message)
#        return pred
    return render_template('index.html', prediction = pred)

if __name__ == '__main__':
    app.run(debug=True)
