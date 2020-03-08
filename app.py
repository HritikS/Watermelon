from flask import Flask, render_template, request
import urllib3.request, json 
import requests
from Depth import build_model, get
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

model = build_model()
model.load_weights('weights.h5')

@app.route('/', methods=['POST'])
def form():
    text = request.form['text']
    text = (pd.DataFrame([[float(i) for i in text.split()]], columns=['Area of Distribution (0-60cm2) ', 'No of Particles (25-300)']) - get()[0]) / get()[1]
    ans = str(model.predict(text))[2:-2]
    return render_template('index.html', ans=ans)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port='5000', debug=True)