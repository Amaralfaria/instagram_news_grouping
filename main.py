from repositories.post_repository import get_all_posts, get_n_posts
from services.pre_processors import normalize, stem
from services.metrics import silhouette_method_best_k, silhouette_metric, get_top_terms_clusters
from services.feature_extractors import extract_count_features, extract_tfidf_features
from services.clusters import cluster_kmeans, cluster_dbscan, cluster_hdbscan
from services.decomposition import lsa_decomposition

if __name__ == "__main__":
    posts = get_n_posts(5000)
    normalized_texts = [stem(normalize(post["text"])) for post in posts]

    X, vectorizer = extract_tfidf_features(normalized_texts)
    X_lsa, svd = lsa_decomposition(X, 200)
    labels, kmeans = cluster_kmeans(X_lsa, 500)
    # labels, dbscan = cluster_hdbscan(X_lsa, 3, "cosine")
    silhouette = silhouette_metric(X_lsa, labels)
    print(silhouette)
    top_words = get_top_terms_clusters(kmeans, X, vectorizer, 10, svd)
    for words in top_words:
        print(", ".join(words))
    

    
