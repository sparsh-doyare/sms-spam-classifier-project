import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- NLTK setup (Render-safe) ---
REQUIRED = ["punkt", "punkt_tab", "stopwords"]
for r in REQUIRED:
    try:
        nltk.data.find(
            f"tokenizers/{r}" if "punkt" in r else f"corpora/{r}"
        )
    except LookupError:
        nltk.download(r)

# --- Globals ---
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [i for i in tokens if i.isalnum()]
    tokens = [i for i in tokens if i not in STOPWORDS and i not in string.punctuation]
    tokens = [ps.stem(i) for i in tokens]

    return " ".join(tokens)

# --- Load model ---
tfidf = pickle.load(open("./vectorizer.pkl", "rb"))
model = pickle.load(open("./model.pkl", "rb"))

# --- UI ---
st.title("EMAIL/SMS SPAM CLASSIFIER")

input_sms = st.text_input("ENTER THE MESSAGE")
if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    st.header("SPAM" if result == 1 else "NOT SPAM")
