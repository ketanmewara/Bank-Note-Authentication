#from flask import Flask, request
import numpy as np 
import pandas as pd
import pickle
#import flasgger
# from flasgger import Swagger
import streamlit as st

# app=Flask(__name__)
# Swagger(app)

pickle_in = open('classifier.pkl', "rb")
classifier = pickle.load(pickle_in)

# @app.route('/')
def welcome():
    return 'Welcome to Bank Note Authentication Web App !'

# @app.route('/predict', methods=['Get'])
def predict_note_authentication(variance,curtosis,skewness,entropy):
  
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    
    """
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction


def main():
    st.title("Bank Note Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Stremlit Bank Note Authenticator ML App </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    variance = st.text_input("Variance", 'Type here')
    curtosis = st.text_input("Curtosis", 'Type_here')
    skewness = st.text_input("Skewness", 'Type_here')
    entropy = st.text_input("Entropy", 'Type_here')
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,curtosis,skewness,entropy)
    st.success('The Output is {}'.format(result))
    # if st.button("About"):
    #     st.text("Let's Earn")
    #     st.text("Build With Streamlit")


if __name__=='__main__':
    main()