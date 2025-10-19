from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import List, Tuple
from scipy.sparse import spmatrix
from numpy import ndarray

def extract_count_features(texts: List[str]) -> Tuple[spmatrix | ndarray , CountVectorizer]:
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def extract_tfidf_features(texts: List[str], min_df: int, max_df: int) -> Tuple[spmatrix, TfidfVectorizer]:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer