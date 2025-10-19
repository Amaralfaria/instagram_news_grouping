import re
import string
import unicodedata
from typing import List
import emoji
import nltk
from nltk.corpus import wordnet_ic, stopwords, wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import RSLPStemmer
import spacy
from nltk.wsd import lesk
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
from enum import Enum

class Relationship(Enum):
    IDENTITY = 1,
    HYPERNYMY = 2
    HYPONYMY = 3,
    MERONYMY = 4
    
nltk.download('stopwords')
nltk.download('rslp')
nltk.download("omw-1.4")
nltk.download("wordnet")
nltk.download('wordnet_ic')

nlp = spacy.load("pt_core_news_md")
stop_words = set(stopwords.words('portuguese'))
stop_words_en = set(stopwords.words('english'))
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

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
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

def remove_tagged_stopwords(tagged_list: list[tuple[str,str,str]]) -> str:
    filtered = [word for word in tagged_list if word[2] not in stop_words]
    return filtered

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
    text =  remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_stopwords(text)
    text=  remove_numbers(text)
    text = remove_accentuation(text)
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

def get_synonyms(word):
    synonyms = set()
    synsets = wn.synsets(word)
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma)
    return sorted(synonyms)  # <- ordena alfabeticamente para consistência

def replace_synonyms(text):
    tokens = text.split()
    processed_tokens = []
    for word in tokens:
        synsets = wn.synsets(word, lang='por')
        if synsets:
            syn = wn.synset(synsets[0].name()).lemma_names('por')[0]
            processed_tokens.append(syn)
        else:
            processed_tokens.append(word)
    
    return " ".join(processed_tokens)

def tag(text: str) -> list[tuple[str, str, str]]:
    doc = nlp(text)
    res = [(token.text, token.pos_, token.lemma_) for token in doc if token.text not in stop_words    ]

    return res

def get_most_likely_synsets(sentence: str, words: list[str]):
    sentence_synsets = []
    for word in words:
        # O Lesk é chamado para cada palavra individualmente
        synset = lesk(sentence, word, pos='n', lang="por")
        if synset:
            sentence_synsets.append(synset)
    
    return sentence_synsets

def pairwise_jcn_similarity(synsets1, synsets2):
    num_synsets1 = len(synsets1)
    num_synsets2 = len(synsets2)
    similarity_matrix = np.zeros((num_synsets1, num_synsets2))

    for i, s1 in enumerate(synsets1):
        for j, s2 in enumerate(synsets2):
            if s1.pos() == s2.pos():
                try:
                    similarity = s1.jcn_similarity(s2, brown_ic)
                    
                    if similarity is not None and similarity > 1:
                        similarity = 1.0
                    
                    similarity_matrix[i, j] = similarity if similarity is not None else 0.0

                except Exception as e:
                    similarity_matrix[i, j] = 0.0
            else:
                similarity_matrix[i, j] = 0.0

    return similarity_matrix

def calculate_sentence_similarity(similarity_matrix, tokens_N, tokens_M):
    cost_matrix = -np.array(similarity_matrix)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    match_value = similarity_matrix[row_ind, col_ind].sum()

    len_N = len(tokens_N)
    len_M = len(tokens_M)

    if (len_N + len_M) == 0:
        return 0.0
    
    final_similarity_score = (2 * match_value) / (len_N + len_M)
    
    return final_similarity_score

# section - lexical chains

def get_des(sense):
    lemmas = [lemma.name() for lemma in sense.lemmas()]
    gloss = get_gloss(sense)
    hypernyms = [get_gloss(s) for s in sense.hypernyms()]
    hyponyms = [get_gloss(s) for s in  sense.hyponyms()]
    meronyms = [get_gloss(s) for s in sense.part_meronyms()]
    holonyms = [get_gloss(s) for s in sense.part_holonyms()]
    atributos = [get_gloss(s) for s in sense.attributes()]
    synsets_similares = [get_gloss(s) for s in sense.similar_tos()]
    synsets_ver_tambem = [get_gloss(s) for s in sense.also_sees()]

    final_set = lemmas + [gloss] + hypernyms + hyponyms + meronyms + holonyms + atributos + synsets_similares + synsets_ver_tambem
    texto = " ".join(final_set)
    tokens = texto.split()
    filtered = [word for word in tokens if word not in stop_words_en]
    return ' '.join(filtered)
    
    
def get_gloss(sense):
    gloss = sense.definition()
    for example in sense.examples():
        gloss = gloss + example
    return gloss

def similarity_custom_lexical_chains(n1_sense, n2_sense):
    # return n1_sense.jcn_similarity(n2_sense, brown_ic)
    # return n1_sense.res_similarity(n2_sense, brown_ic)
    return n1_sense.wup_similarity(n2_sense)

def wsd_custom_lexical_chains(noun: str, nouns: list[str]):
    return lesk(" ".join(nouns), noun, pos='n', lang="por")
    noun_set = set(nouns)
    n_synset = wn.synsets(noun, lang='por')
    
    most_likely_n1_sense = None
    n1_sense_score = 0
    # print(nouns)
    # print(n_synset)

    for n1_sense in n_synset[:3]:
        n1_current_score = 0
        for n2 in noun_set:
            n2_synset = wn.synsets(n2, lang='por')
            n2_sense_score = 0
            for n2_sense in n2_synset[:3]:
                similarity = similarity_custom_lexical_chains(n1_sense, n2_sense)
                n2_sense_score = max(similarity, n2_sense_score)
            n1_current_score += n2_sense_score
        if n1_current_score > n1_sense_score:
            n1_sense_score = n1_current_score
            most_likely_n1_sense = n1_sense

    return most_likely_n1_sense
        
def get_weighted_words(senses: list[str]):
    word_map = {}
    for sense in senses:
        if sense is None:
            continue
        if sense not in word_map:
            word_map[sense] = 1
        else:
            word_map[sense] += 1

    return word_map

def get_all_meronyms(synset):
    part_meronyms = set(synset.part_meronyms())
    substance_meronyms = set(synset.substance_meronyms())
    member_meronyms = set(synset.member_meronyms())
    
    return part_meronyms.union(substance_meronyms).union(member_meronyms)

def find_first_relation(s1, s2):
    similarity = similarity_custom_lexical_chains(s1, s2)
    if (similarity > 0.70):
        return similarity
    return None

    if s1 == s2:
        return Relationship.IDENTITY

    if s1 in s2.hypernyms():
        return Relationship.HYPERNYMY
    
    if s2 in s1.hypernyms():
        return Relationship.HYPERNYMY

    s2_meronyms = get_all_meronyms(s2)
    if s1 in s2_meronyms:
        return Relationship.MERONYMY
    
    s1_meronyms = get_all_meronyms(s1)
    if s2 in s1_meronyms:
        return Relationship.MERONYMY

    return None



def build_lexical_chains_graph(weighted_senses):
    G = nx.Graph()
    for sense in weighted_senses:
        G.add_node(sense, freq=weighted_senses[sense])

    for s1 in weighted_senses:
        for s2 in weighted_senses:
            if s1 == s2:
                continue
            relationship = find_first_relation(s1, s2)
            if relationship is not None and not G.has_edge(s1, s2):
                G.add_edge(s1, s2, relationship=relationship)
                G.add_edge(s2, s1, relationship=relationship)

    return G

def get_connected_components(graph: nx.Graph) -> list[nx.Graph]:
    components = nx.connected_components(graph)
    return [graph.subgraph(c) for c in components]

def fill_concepts_score(subgraph: nx.Graph):
    for n1, n1_data in subgraph.nodes(data=True):
        component_score = n1_data["freq"]
        for n2, n2_data in subgraph.nodes(data=True):
            if not subgraph.has_edge(n1, n2):
                continue
            component_score += n2_data["freq"]*subgraph.get_edge_data(n1,n2)["relationship"]
        # print(component_score)
        # print("\nFormatando os dados das arestas em um loop:")
        # for u, v, dados in subgraph.edges(n1, data=True):
        #     print(f"  - Aresta: ({u}, {v})")
        #     print(f"    Atributos: {dados}")
        n1_data["score"] = component_score

def get_lexical_chain_score(subgraph: nx.Graph):
    score = 0
    for n, n_data in subgraph.nodes(data=True):
        score += n_data["score"]

    return score

def get_lexical_chains_threshold(scores: list[float], alpha = 1.5) -> list[nx.Graph]:
    threshold = 0
    for score in scores:
        threshold += score
    threshold /= len(scores)
    threshold *= alpha

    return threshold

def graph_is_relevant(subgraph: nx.Graph, threshhold):
    return subgraph.graph["score"] >= threshhold




                

                









