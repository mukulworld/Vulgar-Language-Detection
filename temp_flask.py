from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
 
app = Flask(__name__)  
 
model = pickle.load(open('model.pkl','rb'))




@app.route('/')
def go(b):
    a = [word.lower() for word in b]
    pred = model.predict(a)
    return pred



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods =["GET", "POST"])
def predict():
    if request.method == "POST":
       # getting input with name = data in HTML form
       data = request.form.get("data")
       ans = go(data)
       return ans
    return render_template("form.html")
 
if __name__=='__main__':
   app.run()