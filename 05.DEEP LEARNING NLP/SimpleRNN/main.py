import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model


#Load IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

#Load pre-trained model with Relu activation

model = load_model('model.h5')


#Helper function 
#function to decode review

def decode_review(encode_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encode_review])

#function to preocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

import streamlit as st

st.title('IMDB MOVIE REVIEW SENTIMENT ANALYSIS')
st.write('Enter a Movie review to Classify it as positive or negative: ')

#User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    #Make Prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    #Display result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')

else:
    st.write('Please Enter a Movie Review') 















