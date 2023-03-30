import streamlit as st
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import regex as re
from bs4 import BeautifulSoup
from io import StringIO
import numpy as np
import pickle
import nltk
nltk.download('punkt')

#from ipynb.fs.full.FoodReviewSentimentAnalysis import preprocess_review

# Required functions

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def stemming(text):
    snowball = SnowballStemmer(language='english')
    store_list=[]
    for token in word_tokenize(text):
        store_list.append(snowball.stem(token))
    return ' '.join(store_list)

def preprocess_review(text,stopwords):
  # Combining all the above stundents 
  preprocessed_text = []
  # tqdm is for printing the status bar
  for sentance in text:
      sentance = re.sub(r"http\S+", "", sentance)
      sentance = BeautifulSoup(sentance, "html.parser").get_text()
      sentance = decontracted(sentance)
      sentance = re.sub("\S*\d\S*", "", sentance).strip()
      sentance = re.sub('[^A-Za-z]+', ' ', sentance)
      
      # https://gist.github.com/sebleier/554280
      temp_sent = ()
      for e in sentance.split():
          if e.lower() == "ok":
            temp_sent += ("okay",)
          elif (e.lower() not in stopwords) & (len(e)>2):
            temp_sent += (e.lower(),)
      sentence = " ".join(temp_sent)
      # Stemming 
      preprocessed_text.append(stemming(sentence.strip()))
  
  return preprocessed_text


# Required variables
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
# Load vecorizer to transform
with open('text_vectorizer_max.pkl', 'rb') as vect_file:
      
    # Call load method to deserialze
    tf_idf_vect = pickle.load(vect_file)


# Load normalizer to transform
with open('text_normalizer_max.pkl', 'rb') as norm_file:
      
    # Call load method to deserialze
    tfIdf_norm = pickle.load(norm_file)

# Load the trained model to predict
with open('sentimentAnalysis_model_max.pkl', 'rb') as model_file:
      
    # Call load method to deserialze
    NbClfTfIdf = pickle.load(model_file)


# Build App
st.title("Food review sentiment analysis")
st.header("Find out percentage of Positive,Neutral and Negative reviews of your product")
st.write("")
st.write("")
st.write("")
st.write("")

st.subheader("Analyze single review")
direct = st.text_input("Write review to know the sentiment")
if st.button("Check Review",key="single"):
    prep =  preprocess_review([direct],stopwords)
    vect = tf_idf_vect.transform(prep)
    norm = tfIdf_norm.transform(vect.toarray())
    direct_op = NbClfTfIdf.predict(norm)[0]
    if direct_op == 2:
      st.success("Positive Review")
    elif direct_op == 1:
      st.warning("Neutral Review")
    else:
      st.error("Negative Review") 
     
st.write("")
st.write("")
st.write("")
st.write("")
st.subheader("Analyze multiple reviews")
sentiment_summary = []

files = st.file_uploader("Choose a text file or multiple text files.",accept_multiple_files=True)
if st.button("Check Review",key="multiple"):
  for uploaded_file in files:
      if uploaded_file is not None:
          for line in uploaded_file:
            review = line.decode()
            prep =  preprocess_review([review],stopwords)
            vect = tf_idf_vect.transform(prep)
            norm = tfIdf_norm.transform(vect.toarray())
            output = NbClfTfIdf.predict(norm)[0]
            sentiment_summary.append(output)
            

  st.markdown("**Review's Sentiment Summary**")
  senti = np.asarray(sentiment_summary)
  total_review_count = len(sentiment_summary)
  st.write("Total review count: {}".format(total_review_count))
  pos_percent = round(np.count_nonzero(senti == 2)/total_review_count,3)
  neu_percent = round(np.count_nonzero(senti == 1)/total_review_count,3)
  neg_percent = round(np.count_nonzero(senti == 0)/total_review_count,3)
  #st.write("Positive reviews : {} %".format(pos_percent*100))
  #st.write("Neutral reviews : {} %".format(neu_percent*100))
  #st.write("Negative reviews : {} %".format(neg_percent*100))
  col1, col2, col3 = st.columns(3)
  col1.metric("Positive", "{}%".format(pos_percent*100))
  col2.metric("Neutral", "{}%".format(neu_percent*100))
  col3.metric("Negative", "{}%".format(neg_percent*100))
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.subheader("Click for more information")
# Source: https://stackoverflow.com/questions/74003574/how-to-create-a-button-with-hyperlink-in-streamlit
url = 'https://github.com/praathapj/FoodReviewSentimentAnalysis'

st.markdown(f'''
<a href={url}><button style="background-color:GreenYellow;">GitHub</button></a>
''',unsafe_allow_html=True)

