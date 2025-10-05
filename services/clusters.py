from typing import List
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN 
import numpy as np
from scipy.sparse import spmatrix
from typing import Tuple, Union

def cluster_kmeans(X: np.ndarray | spmatrix, n_clusters: int) -> Tuple[np.ndarray, KMeans]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return (kmeans.labels_, kmeans)

def cluster_hierarchical(X: np.ndarray | spmatrix, n_clusters: int, linkage: str) -> Tuple[np.ndarray, AgglomerativeClustering]:
    hier = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    hier.fit(X)
    return hier.labels_

def cluster_dbscan(
    X: np.ndarray | spmatrix,
    eps: float,
    min_samples: int,
    metric: str
) -> Tuple[np.ndarray, DBSCAN]:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(X)
    return dbscan.labels_, dbscan

def cluster_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    metric: str,
    min_samples: int = None,
) -> Tuple[np.ndarray, HDBSCAN]:
    """
    Realiza clustering usando HDBSCAN.
    
    Parâmetros:
    - X: matriz de features (TF-IDF, LSA, CountVectorizer)
    - min_cluster_size: tamanho mínimo de cada cluster
    - min_samples: número mínimo de pontos para formar núcleo (opcional)
    - metric: métrica de distância (ex: 'euclidean', 'cosine')
    
    Retorna:
    - labels: array com rótulos do cluster (-1 = ruído)
    - modelo HDBSCAN treinado
    """
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric
    )
    
    labels = model.fit_predict(X)
    return labels, model