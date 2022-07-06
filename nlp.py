import string

import nltk
from nltk.stem import WordNetLemmatizer
import joblib

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

vectorizer = joblib.load("vectorizer.pkl")
svd = joblib.load("svd.pkl")
mlb_target = joblib.load("mlb_target.pkl")
model = joblib.load("model.pkl")


def preprocess_single_document(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    to_lower = punctuationfree.lower()
    tokenizer = nltk.word_tokenize(to_lower)
    remove_sw = [i for i in tokenizer if i not in stopwords]
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in remove_sw]
    x_str = " ".join(lemm_text)
    X = vectorizer.transform([x_str])
    x_reduced = svd.transform(X)
    return x_reduced


def get_tag(text):
    x = preprocess_single_document(text)
    pred_proba = model.predict_proba(x)
    tag = [mlb_target.classes_[item] for item in pred_proba.argmax(axis=1)]
    return {"tag": tag[0]}


