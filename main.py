import re
import json
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI
from pydantic import BaseModel
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences

app = FastAPI()

# text preprocessing
def process_text(text):
    text=text.lower()
    text=re.sub("<br ?/?>", " ", text)
    text=re.sub("[^\w\s]", " ", text)
    return text


# lemmatization
wnl = WordNetLemmatizer()

def lang_process(text):
    tokens = [word for word in word_tokenize(text)]
    stems = [wnl.lemmatize(word, pos='v') for word in tokens]
    st=" ".join(stems)
    
        
    return st

# model import
senti_model = tf.keras.models.load_model(r"model_building\saved_model\my_model")

# tokenizer import
with open(r"model_building\tokenizer.json") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# query processor
class QueryString(BaseModel):
    query: str


# landing api
@app.get("/")
def root():
    return {"message": "Hello World"}


# sentiment analysis
@app.post("/sentiment_analysis")
def predict_sentiment(text):

    text = process_text(text)
    clean_text = lang_process(text)

    sentence = [clean_text]
    sequences = tokenizer.texts_to_sequences(sentence)

    padded = pad_sequences(sequences, padding='post', maxlen=200)
    
    prediction = senti_model.predict(padded)
    
    pred_labels = []
    for i in prediction:
        if i >= 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    senti_labels=[]            
    for i in range(len(sentence)):
        if pred_labels[i] == 1:
            senti_labels.append('Positive')
        else:
            senti_labels.append('Negative')
    
    pred = prediction.tolist()

    return {
        "sentiment": senti_labels[0],
        "confidence": round(pred[0][0], 3)
        }