#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:21:35 2023

@author: nirav-ubuntu
"""
import streamlit as st
import pickle
from model import predict_sentiment

# Load the model
model = pickle.load(open('sentiment_model.pkl','rb'))

# Load the tokenizer
tokenizer = pickle.load(open('sentiment_token.pkl','rb'))

# Create the Streamlit app
def main():
    st.title("Movie Review Sentiment Prediction")
    
    # Get user input
    review = st.text_area("Enter a movie review:")
    
    if st.button("Predict"):
        # Perform prediction when the user clicks the "Predict" button
        percent, sentiment = predict_sentiment(review,model,tokenizer)
        
        # Display the prediction
        if sentiment == "POSITIVE":
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")

if __name__ == '__main__':
    main()
