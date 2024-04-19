import streamlit as st
from azureml.core import Workspace
from azureml.core.model import Model
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
import re
import joblib

# nltk.download('wordnet')

ws = Workspace.from_config()
# model2 = Model(ws, "test-model1")
model_path = Model.get_model_path("azureml://locations/centralindia/workspaces/25fa9527-518b-4d4d-b87a-71320f5df619/models/test-model1/versions/1")
model2 = joblib.load(model_path)

def sentiment_emoji(sentiment):
    if sentiment == 'positive':
        return "😊"
    elif sentiment == 'negative':
        return "😞"
    else:
        return ""

def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        tweet = re.sub(urlPattern,' URL',tweet)
        
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])      
        
        tweet = re.sub(userPattern,' USER', tweet)      
        
        tweet = re.sub(alphaPattern, " ", tweet)
        
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            
            if len(word)>1:
                
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

def load_models():    
    
    file = open('./vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('./Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    print(sentiment)
    
    return "Negative" if sentiment[0] == 0 else "Positive"

def main():
    
    vectorizer, model1 = load_models()
    
    st.title("Model Deployment with Streamlit")

    feature1 = st.text_input("Feature 1", placeholder="Enter the Tweet")

    # Predict function
    if st.button("Predict"):
        prediction = predict(vectorizer, model1, [feature1])
        
        sentiment = prediction.lower()
        
        st.markdown(f"<p style='font-size: 24px;'>Predicted sentiment: {sentiment} {sentiment_emoji(sentiment)}</p>", unsafe_allow_html=True)

        # Display colored result
        if sentiment == 'positive':
            st.success("Positive sentiment detected!")
        elif sentiment == 'negative':
            st.error("Negative sentiment detected.")
        # st.write(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()