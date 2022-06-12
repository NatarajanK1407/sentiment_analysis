# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import streamlit as st
df = pickle.load(open('./model.pkl', 'rb'))

st.header("Type the sentence to be analyzed......")
text = st.text_input(' ')


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeClassifier  # RidgeClassifier
from sklearn.semi_supervised import SelfTrainingClassifier  # SelfTrainingClassifier

# Data Preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    df.content, df.sentiment, test_size=0.7, random_state=25, shuffle=True)
# NLP system entegration to Data
X_CountVectorizer = CountVectorizer(stop_words='english')

X_train_counts = X_CountVectorizer.fit_transform(X_train)

X_TfidfTransformer = TfidfTransformer()

X_train_tfidf = X_TfidfTransformer.fit_transform(X_train_counts)

# Model Creating
model_semi = SelfTrainingClassifier(RidgeClassifier())

model_semi.fit(X_train_tfidf, y_train)

# Data of Prediction

text = [text]

text_counts = X_CountVectorizer.transform(text)

# Prediction Processing
prediction = model_semi.predict(text_counts)

st.write("\n\n The prediction in the sentece(s) is : ",prediction[0])
f"Prediction is {prediction[0]}"

