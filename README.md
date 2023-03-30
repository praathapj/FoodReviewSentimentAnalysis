# Food Review Sentiment Analysis

## **Business Problem**

A food product will have multiple reviews from multiple websites with rating or without rating but only text. As a product owner I want to understand the count or percentage of total Positive, Neutral and Negative reviews for the product.

## Overview
An ML model which can analyzing hundreds of food reviews and return the percentage of Positive, Neutral and Negative reviews.

* **Deployed web app :**https://praathapj-foodreviewsentimentana-foodreviewsentiment-app-e1nreg.streamlit.app/

## Data Collection
Real world data obtained from kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews


## **Data Cleaning and Preprocessing**
* Removed non realstic data and joined review summery and text as one single feature.
* Aim was to predict Sentiment hence processed rating 5,4 as Positive, 3 as neutral and 2,1 as Negative.
* Considered only the text data to predict sentiment.

## Data Analysis
* Highly imbalanced dataset, positive reviews were more than twice compared to other two class.
* Performed random Undersampling and oversampling with median samples of distribution.

## Features Engineering or Vectoriztion
* Performed Bag Of Words(BOW), n gram BOW, Binary BOW, TF-IDF, Word 2 Vector(W2V), avg W2V and TF-IDF W2V
* All the vector were scaled between 0 to 1

## Modelling
Very high dimensional data approx 5000 features, hence Naive bayese was used as it works very well and performance constraints.

## Model Evaluation
* Evaluation metric is multi class log loss and confusino matrics.
* Random model was used to baseline the max log loss: 0.69
* TF-IDF gave the least log loss 0.58

## Improvement
* Use more balanced data than sampling and try with other algorithms which requires huge computation.
