
from repositories.post_repository import get_all_posts, get_n_posts
from services.pre_processors import *
from services.metrics import silhouette_metric, get_top_terms_clusters, test_max_df, test_min_df, test_svd, test_n_clusters   
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN , SpectralClustering, AffinityPropagation
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from services.clusters import *
from nltk.corpus import wordnet_ic, stopwords, wordnet


regex_numeros = re.compile(r'\b\w*\d\w*\b')

def get_wanted_synsets(original_text, tagged_text):
    wanted_tags = ["NOUN"]
    wanted_words = [token[2] for token in tagged_text if token[1] in wanted_tags and not regex_numeros.search(token[0])]

    return get_most_likely_synsets((original_text), wanted_words)




if __name__ == "__main__":
    posts = pd.DataFrame(get_n_posts(500))
    normalized_texts = [(remove_html_tags(remove_urls(remove_call_to_action(remove_emojis(post))))) for post in posts['text'].tolist()]

    print(normalized_texts[0])

    tuple_texts = [(text, tag(text)) for text in normalized_texts]
    final_output = [(text[0], text[1], get_wanted_synsets(text[0], text[1])) for text in tuple_texts]
    documents_similarity_matrix = np.zeros((len(posts), len(posts)))

    for i, line1 in enumerate(final_output):
        for j, line2 in enumerate(final_output):
            jcn_matrix = pairwise_jcn_similarity(line1[2], line2[2])
            sentence_similarity = calculate_sentence_similarity(jcn_matrix, line1[1], line2[1])
            documents_similarity_matrix[i, j] = sentence_similarity

    # spectral_model = SpectralClustering(
    #     n_clusters=10,
    #     affinity='precomputed',
    #     random_state=42
    # )

    # labels = spectral_model.fit_predict(documents_similarity_matrix)

    ap_stable = AffinityPropagation(affinity='precomputed', 
                                damping=0.5,  # Aumentamos o damping para 0.9
                                random_state=0)

    ap_stable.fit(documents_similarity_matrix)
    labels = ap_stable.labels_

    posts["cluster"] = labels
    df_sorted = posts.sort_values(by='cluster')

    final_output = df_sorted[['text', 'cluster']]

    final_output.to_csv('clusters_ordenados.csv', index=False)


    











