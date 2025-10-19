from repositories.post_repository import get_all_posts, get_n_posts
from services.pre_processors import normalize, stem, lemmatize
from services.metrics import silhouette_metric, get_top_terms_clusters, test_max_df, test_min_df, test_svd, test_n_clusters
from services.feature_extractors import extract_count_features, extract_tfidf_features
from services.clusters import cluster_kmeans, cluster_dbscan, cluster_hdbscan
from services.decomposition import lsa_decomposition
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN 
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

if __name__ == "__main__":
    posts = pd.DataFrame(get_n_posts(5000))
    normalized_texts = [stem(normalize(post)) for post in posts['text'].tolist()]

    silhouette_score_values = []
    distortions = []
    n_max_df_options = [100, 500, 1000, 2000, 3000, 4000, 4999] # melhores 1000 para k mais baixo, 500 para k mais alto
    n_min_df_options = [2, 5, 10, 15, 20, 25] # melhores 15 para k mais alto, 20 para mais baixo
    n_components_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # 20 mais alto, 10 mais baixo
    n_clusters_options = range(2, 30) # 23

    vectorizer = TfidfVectorizer(max_df=500, min_df=15)
    X_TFIDF = vectorizer.fit_transform(normalized_texts)
    lsa_model = TruncatedSVD(n_components=23, random_state=42)
    X_lsa = lsa_model.fit_transform(X_TFIDF)
    X_lsa = Normalizer(copy=False).fit_transform(X_lsa)
    kmeans = KMeans(n_clusters=23, random_state=42)
    # similarity_matrix = cosine_similarity(X_TFIDF)
    distance_matrix = cosine_distances(X_lsa)
    kmeans.fit(X_lsa)
    cluster_silhouette_score = silhouette_score(distance_matrix, kmeans.labels_.ravel(), metric="precomputed")
    print(cluster_silhouette_score)
    posts['cluster'] = kmeans.labels_
    df_sorted = posts.sort_values(by='cluster')

    final_output = df_sorted[['text', 'cluster']]

    final_output.to_csv('clusters_ordenados.csv', index=False)





    # test_n_clusters(normalized_texts, 15, 500, 20, n_clusters_options)
    
    

            

    
    # lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)
    # X_lsa = lsa_model.fit_transform(X)
    # X_lsa = Normalizer(copy=False).fit_transform(X_lsa)

    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # kmeans.fit(X)
    # (kmeans.labels_, kmeans)

    # labels, dbscan = cluster_hdbscan(X_lsa, 3, "cosine")
    # silhouette = silhouette_metric(X_lsa, labels)
    # print(silhouette)
    top_words = get_top_terms_clusters(kmeans, X, vectorizer, 10, svd)
    # for words in top_words:
    #     print(", ".join(words))
    

    
