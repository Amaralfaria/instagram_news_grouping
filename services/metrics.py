from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from typing import Tuple
from typing import Union
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import Normalizer

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

def test_max_df(normalized_texts: list[str], max_df_options: list[float], n_clusters: int):
    current_silhouette_list = []
    current_distortions_list = []
    for n_max_df in max_df_options:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        vectorizer = TfidfVectorizer(max_df=n_max_df, min_df=0.0)
        X_TFIDF = vectorizer.fit_transform(normalized_texts)
        # similarity_matrix = cosine_similarity(X_TFIDF)
        distance_matrix = cosine_distances(X_TFIDF)
        kmeans.fit(X_TFIDF)
        current_distortions_list.append(kmeans.inertia_)
        cluster_silhouette_score = silhouette_score(distance_matrix, kmeans.labels_.ravel(), metric="precomputed")
        current_silhouette_list.append(cluster_silhouette_score)


    output_dir = f'experiments/kmeans-tfidf-svd-distances/max_df/clusters_{n_clusters}'
    os.makedirs(output_dir, exist_ok=True)

    plot1 = plt.figure(f'{n_clusters}_{n_max_df}1', figsize=(16, 8))
    plt.title(f'Sum of Squared Errors vs max_df - K = {n_clusters}')
    plt.xlabel('max_df')
    plt.ylabel('SSE')
    plt.plot(max_df_options, np.array(current_distortions_list).ravel())

    plt.savefig(f'{output_dir}/sse_vs_n_max_df.png')
    plt.close()
    plot2 = plt.figure(f'{n_clusters}_{n_max_df}2', figsize=(16, 8))
    
    plt.title(f'Silhouette Score vs max_df - K = {n_clusters}')
    plt.xlabel('max_df')
    plt.ylabel('Silhouette score')
    plt.plot(max_df_options, np.array(current_silhouette_list).ravel())

    plt.savefig(f'{output_dir}/silhouette_vs_n_max_df.png')
    plt.close()

def test_min_df(normalized_texts: list[str], min_df_options: list[int], n_clusters: int, max_df: float):
    current_silhouette_list = []
    current_distortions_list = []
    for n_min_df in min_df_options:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=n_min_df)
        X_TFIDF = vectorizer.fit_transform(normalized_texts)
        # similarity_matrix = cosine_similarity(X_TFIDF)
        distance_matrix = cosine_distances(X_TFIDF)
        kmeans.fit(X_TFIDF)
        current_distortions_list.append(kmeans.inertia_)
        cluster_silhouette_score = silhouette_score(distance_matrix, kmeans.labels_.ravel(), metric="precomputed")
        current_silhouette_list.append(cluster_silhouette_score)


    output_dir = f'experiments/kmeans-tfidf-svd-distances/min_df/clusters_{n_clusters}'
    os.makedirs(output_dir, exist_ok=True)

    plot1 = plt.figure(f'{n_clusters}_{n_min_df}1', figsize=(16, 8))
    plt.title(f'Sum of Squared Errors vs min_df - K = {n_clusters}')
    plt.xlabel('min_df')
    plt.ylabel('SSE')
    plt.plot(min_df_options, np.array(current_distortions_list).ravel())

    plt.savefig(f'{output_dir}/sse_vs_n_min_df.png')
    plt.close()
    plot2 = plt.figure(f'{n_clusters}_{n_min_df}2', figsize=(16, 8))
    
    plt.title(f'Silhouette Score vs min_df - K = {n_clusters}')
    plt.xlabel('min_df')
    plt.ylabel('Silhouette score')
    plt.plot(min_df_options, np.array(current_silhouette_list).ravel())

    plt.savefig(f'{output_dir}/silhouette_vs_n_min_df.png')
    plt.close()

def test_svd(normalized_texts: list[str], min_df: int, n_clusters: int, max_df: float, n_components_options: list[int]):
    current_silhouette_list = []
    current_distortions_list = []
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    X_TFIDF = vectorizer.fit_transform(normalized_texts)
    for n_components in n_components_options:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
        X_lsa = lsa_model.fit_transform(X_TFIDF)
        X_lsa = Normalizer(copy=False).fit_transform(X_lsa)
        # similarity_matrix = cosine_similarity(X_TFIDF)
        distance_matrix = cosine_distances(X_lsa)
        kmeans.fit(X_lsa)
        current_distortions_list.append(kmeans.inertia_)
        cluster_silhouette_score = silhouette_score(distance_matrix, kmeans.labels_.ravel(), metric="precomputed")
        current_silhouette_list.append(cluster_silhouette_score)


    output_dir = f'experiments/kmeans-tfidf-svd-distances/n_components/clusters_{n_clusters}'
    os.makedirs(output_dir, exist_ok=True)

    plot1 = plt.figure(f'{n_clusters}_{n_components}1', figsize=(16, 8))
    plt.title(f'Sum of Squared Errors vs n_components - K = {n_clusters}')
    plt.xlabel('n_components')
    plt.ylabel('SSE')
    plt.plot(n_components_options, np.array(current_distortions_list).ravel())

    plt.savefig(f'{output_dir}/sse_vs_n_components.png')
    plt.close()
    plot2 = plt.figure(f'{n_clusters}_{n_components}2', figsize=(16, 8))
    
    plt.title(f'Silhouette Score vs n_components - K = {n_clusters}')
    plt.xlabel('n_components')
    plt.ylabel('Silhouette score')
    plt.plot(n_components_options, np.array(current_silhouette_list).ravel())

    plt.savefig(f'{output_dir}/silhouette_vs_n_components.png')
    plt.close()

def test_n_clusters(normalized_texts: list[str], min_df: int, max_df: float, n_components: int, n_clusters_options: list[int]):
    current_silhouette_list = []
    current_distortions_list = []
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    X_TFIDF = vectorizer.fit_transform(normalized_texts)
    lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
    X_lsa = lsa_model.fit_transform(X_TFIDF)
    X_lsa = Normalizer(copy=False).fit_transform(X_lsa)
    for n_clusters in n_clusters_options:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # similarity_matrix = cosine_similarity(X_TFIDF)
        distance_matrix = cosine_distances(X_lsa)
        kmeans.fit(X_lsa)
        current_distortions_list.append(kmeans.inertia_)
        cluster_silhouette_score = silhouette_score(distance_matrix, kmeans.labels_.ravel(), metric="precomputed")
        current_silhouette_list.append(cluster_silhouette_score)


    output_dir = f'experiments/kmeans-tfidf-svd-distances/n_clusters/clusters_{n_clusters}'
    os.makedirs(output_dir, exist_ok=True)

    plot1 = plt.figure(f'{n_clusters}_{n_components}1', figsize=(16, 8))
    plt.title(f'Sum of Squared Errors vs n_clusters')
    plt.xlabel('n_clusters')
    plt.ylabel('SSE')
    plt.plot(n_clusters_options, np.array(current_distortions_list).ravel())

    plt.savefig(f'{output_dir}/sse_vs_n_clusters.png')
    plt.close()
    plot2 = plt.figure(f'{n_clusters}_{n_clusters}2', figsize=(16, 8))
    
    plt.title(f'Silhouette Score vs n_clusters')
    plt.xlabel('n_cluesters')
    plt.ylabel('Silhouette score')
    plt.plot(n_clusters_options, np.array(current_silhouette_list).ravel())

    plt.savefig(f'{output_dir}/silhouette_vs_n_clusters.png')
    plt.close()


