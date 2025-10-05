from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from typing import Tuple
from typing import Union
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 

def silhouette_method_best_k(X: np.ndarray, k_range: Tuple[int, int]) -> int:
    silhouette_scores = []
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    best_k = np.argmax(silhouette_scores) + k_range[0]
    
    return best_k

def silhouette_metric(X: np.ndarray, labels: Union[np.ndarray, list]) -> float:
    if len(set(labels)) > 1:  # precisa de pelo menos 2 clusters
        return silhouette_score(X, labels)
    else:
        return float('nan')


def davies_bouldin_metric(X: np.ndarray, labels: Union[np.ndarray, list]) -> float:
    if len(set(labels)) > 1:
        return davies_bouldin_score(X, labels)
    else:
        return float('nan')


def calinski_harabasz_metric(X: np.ndarray, labels: Union[np.ndarray, list]) -> float:
    if len(set(labels)) > 1:
        return calinski_harabasz_score(X, labels)
    else:
        return float('nan')

def count_elements_per_cluster(labels: np.ndarray, include_noise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    unique_labels = np.unique(labels)
    
    if not include_noise:
        unique_labels = unique_labels[unique_labels != -1]
    
    counts = np.array([np.sum(labels == lbl) for lbl in unique_labels])
    return unique_labels, counts

def get_top_terms_clusters(
    model: Union[KMeans, DBSCAN],
    X,
    vectorizer: Union[TfidfVectorizer, CountVectorizer],
    top_n: int = 10,
    lsa_model: TruncatedSVD = None,
    include_noise: bool = False
) -> list[list[str]]:
    terms = vectorizer.get_feature_names_out()
    
    labels = model.labels_
    
    unique_labels = np.unique(labels)
    if not include_noise:
        unique_labels = unique_labels[unique_labels != -1]
    
    if isinstance(model, KMeans) and lsa_model is None:
        cluster_centers = model.cluster_centers_
    else:
        cluster_centers = []
        for lbl in unique_labels:
            points = X[labels == lbl]
            if lsa_model is not None:
                points_lsa = lsa_model.transform(points)
                center_lsa = points_lsa.mean(axis=0).reshape(1, -1)
                center_orig = lsa_model.inverse_transform(center_lsa)[0]
            else:
                center_orig = points.mean(axis=0).A1 if hasattr(points, "A1") else points.mean(axis=0)
            cluster_centers.append(center_orig)
        cluster_centers = np.array(cluster_centers)
    
    order_centroids = cluster_centers.argsort()[:, ::-1]
    
    top_terms_clusters = []
    for i in range(len(unique_labels)):
        top_terms = [terms[ind] for ind in order_centroids[i, :top_n]]
        top_terms_clusters.append(top_terms)
    
    return top_terms_clusters