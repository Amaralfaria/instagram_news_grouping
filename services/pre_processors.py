import re
import string
import unicodedata
from typing import List
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import spacy

nltk.download('stopwords')
nltk.download('rslp')

nlp = spacy.load("pt_core_news_lg")

triggers = [
    "leia a matéria completa",
    "confira a matéria completa",
    "veja a matéria completa",
    "saiba mais",
    "mais informações",
    "clique no link",
    "acesse o link",
    "confira no link",
    "veja também",
    "curta e compartilhe",
    "marque um amigo",
    "comente abaixo",
    "envie para alguém",
    "compartilhe com amigos",
    "clique no link da bio",
    "link na bio",
    "mais detalhes nos stories",
    "confira nos stories",
    "veja nos stories",
    "saiba todos os detalhes",
    "não perca",
    "imperdível",
    "confira agora",
]

def lowercase(text: str) -> str:
    return text.lower()

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace="")

def remove_special_characters(text: str) -> str:
    return re.sub(r'[^A-Za-zÀ-ÿ0-9\s]', '', text)

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words('portuguese'))
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

def remove_urls(text: str) -> str:
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_html_tags(text: str) -> str:
    return re.sub(r'<.*?>', '', text)

def remove_numbers(text: str) -> str:
    return re.sub(r'\b\w*\d\w*\b', '', text)

def stem(text: str) -> str:
    stemmer = RSLPStemmer()
    tokens = text.split()
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

def lemmatize(text: str) -> str:
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def normalize(text: str) -> str:
    text = lowercase(text)
    text = remove_emojis(text)
    text = remove_call_to_action(text)
    text=  remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_accentuation(text)
    text = remove_stopwords(text)
    return text

def remove_accentuation(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def remove_call_to_action(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[➡️👉]", "", text)
    
    triggers_pattern = re.compile("|".join(map(re.escape, triggers)), re.IGNORECASE)
    match = triggers_pattern.search(text)
    if match:
        text = text[:match.start()]
    
    doc = nlp(text)
    cutoff = len(text)
    for sent in doc.sents:
        if any(token.tag_ == "VIMP" for token in sent):
            cutoff = sent.start_char
            break
    
    return text[:cutoff].strip()