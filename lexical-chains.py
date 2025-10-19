from repositories.post_repository import *
from services.pre_processors import *
from nltk.corpus import wordnet_ic, stopwords, wordnet as wn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.cluster import BisectingKMeans
from sklearn.feature_extraction import DictVectorizer
from services.metrics import *

if __name__ == "__main__":
    posts = pd.DataFrame(get_n_posts(1000))
    tagged_texts = [remove_tagged_stopwords(tag(remove_html_tags(remove_urls(remove_call_to_action(remove_emojis(post)))))) for post in posts['text'].tolist()]
    noun_texts = []
    for tagged_text in tagged_texts:
        noun_texts.append([text for text in tagged_text if text[1] == "NOUN"])


    disambiguated_texts = []
    for noun_text in noun_texts:
        text = []
        for token in noun_text:
            text.append((token[0], token[1], token[2], wsd_custom_lexical_chains(token[2], [noun[2] for noun in noun_text])))
        disambiguated_texts.append(text)

    weighted_nouns = []
    for text in disambiguated_texts:
        weighted_nouns.append(get_weighted_words([word[3] for word in text]))

    # print(weighted_nouns)

    lexical_graphs = []
    for text in weighted_nouns:
        lexical_graphs.append(build_lexical_chains_graph(text))

    relevant_lexical_graphs = []
    for i, graph in enumerate(lexical_graphs):
        scored_lexical_chains = []
        components = get_connected_components(graph)
        for c in components:
            fill_concepts_score(c)
        for c in components:
            score = get_lexical_chain_score(c)
            scored_lexical_chains.append((c, score))

        threshold = get_lexical_chains_threshold([chain[1] for chain in scored_lexical_chains])
        relevant_lexical_chains = filter(lambda c: c[1] >= threshold, scored_lexical_chains)
        # print("--- Método 3: Imprimindo arestas com seus dados/atributos ---")
        # print(" ".join([word[0] for word in noun_texts[i]]))
        # for lc in relevant_lexical_chains:
        #     print(lc[1])
        #     arestas_com_dados = lc[0].edges(data=True)
        #     print("Lista de arestas com seus dicionários de atributos:")
        #     print(arestas_com_dados)

        #     print("\nFormatando os dados das arestas em um loop:")
        #     for u, v, dados in lc[0].edges(data=True):
        #         print(f"  - Aresta: ({u}, {v})")
        #         print(f"    Atributos: {dados}")
        #     print("-" * 40 + "\n")
        relevant_lexical_graphs.append(relevant_lexical_chains)

    final_info = []
    for graph in relevant_lexical_graphs:
        doc_dict = {}
        for chain in graph:
            for node, data in chain[0].nodes(data=True):
                doc_dict[node.name()] = data["score"]
        final_info.append(doc_dict)

    # for info in final_info:
    #     print(info)


    dv = DictVectorizer()

    # 2. Crie a matriz
    X_weighted = dv.fit_transform(final_info)

    inertias = []
    k_range = range(3, 50)
    for k in k_range:
    # Criar e treinar o modelo BisectingKMeans
        bisecting_model = BisectingKMeans(
            n_clusters=k, 
            random_state=42,
            n_init=10
        )
        bisecting_model.fit(X_weighted)
        
        # Adicionar a inércia (SSE) do modelo à nossa lista
        inertias.append(bisecting_model.inertia_)

    print(f"Valores de k testados: {list(k_range)}")
    print(f"Valores de Inércia (SSE): {inertias}")

    # --- 3. Plotando o Gráfico do Joelho ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-') # 'bo-' = blue dots and solid line
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia (Soma dos Erros Quadrados)')
    plt.title('Método do Joelho (Elbow Method) usando Bisecting K-Means')
    plt.grid(True)
    plt.savefig('elbow_plot.png')
    print("\nGráfico 'elbow_plot.png' salvo. Analise-o para encontrar o 'joelho'.")


    # bisecting_model = BisectingKMeans(
    #     n_clusters=2, 
    #     random_state=42,
    #     n_init=10
    # )

    # bisecting_model.fit(X_weighted)

    # labels = bisecting_model.labels_

    # posts['cluster'] = labels
    # df_sorted = posts.sort_values(by='cluster')

    # final_output = df_sorted[['text', 'cluster']]

    # final_output.to_csv('clusters_ordenados.csv', index=False)

    # top_words = get_top_terms_clusters(bisecting_model, X_weighted, dv, 10)
    # print(top_words)

    


    # for i, graph in enumerate(lexical_graphs):
        
    #     print(" ".join([word[0] for word in noun_texts[i]]))
    #     print("--- Método 3: Imprimindo arestas com seus dados/atributos ---")
    #     arestas_com_dados = graph.edges(data=True)
    #     print("Lista de arestas com seus dicionários de atributos:")
    #     print(arestas_com_dados)

    #     print("\nFormatando os dados das arestas em um loop:")
    #     for u, v, dados in graph.edges(data=True):
    #         print(f"  - Aresta: ({u}, {v})")
    #         print(f"    Atributos: {dados}")
    #     print("-" * 40 + "\n")

    


