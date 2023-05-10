#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:01:38 2023

@author: nirav-ubuntu
"""

import pickle
from flask import Flask,request,app,jsonify,render_template,url_for
from model import predict_sentiment

app = Flask(__name__)

## Load the model
model = pickle.load(open('sentiment_model.pkl','rb'))

## Load the tokenizer
tokenizer = pickle.load(open('sentiment_token.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['GET','POST'])
def predict_api():
    data = request.form.values()
    review = data[0]
    percent, sentiment = predict_sentiment(review,model,tokenizer)
    response = {'sentiment': sentiment, 'percent': percent}
    return jsonify(response)

@app.route('/predict',methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    review = data[0]
    percent, sentiment = predict_sentiment(review,model,tokenizer)
    return render_template("home.html",review_text=f"\nReview: {review}",sentiment_text=f"\nSentiment: {sentiment}",percent_text=f"\nPercentage: {round(percent*100,2)}%")
    
if __name__=="__main__":
    app.run(debug=True)
  
