from typing import Tuple, Union
import numpy as np
from scipy.sparse import spmatrix
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.preprocessing import Normalizer

def lsa_decomposition(X: Union[spmatrix, np.ndarray], n_topics: int) -> Tuple[np.ndarray, TruncatedSVD]:
    lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)
    X_lsa = lsa_model.fit_transform(X)
    X_lsa = Normalizer(copy=False).fit_transform(X_lsa)
    return X_lsa, lsa_model

def nmf_decomposition(X: Union[spmatrix, np.ndarray], n_topics: int) -> Tuple[np.ndarray, NMF]:
    nmf_model = NMF(n_components=n_topics, random_state=42)
    X_nmf = nmf_model.fit_transform(X)
    return X_nmf, nmf_model

def lda_decomposition(X: Union[spmatrix, np.ndarray], n_topics: int) -> Tuple[np.ndarray, LatentDirichletAllocation]:
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    X_lda = lda_model.fit_transform(X)
    return X_lda, lda_model