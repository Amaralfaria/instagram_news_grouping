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

class SpectralDivideAndMerge:
    """
    Implementação do algoritmo Spectral Divide and Merge para clustering.
    
    Este algoritmo encontra o número de clusters automaticamente através de 
    bisseção recursiva baseada no espectro da matriz Laplaciana.
    """
    def __init__(self, lambda_threshold=0.8, min_cluster_size=2):
        """
        Args:
            lambda_threshold (float): Limiar para o segundo menor autovalor (λ₂). 
                                     Se λ₂ for menor que este valor, o cluster é dividido.
                                     Valores mais altos levam a mais divisões.
            min_cluster_size (int): Tamanho mínimo para que um cluster seja considerado para divisão.
        """
        self.lambda_threshold = lambda_threshold
        self.min_cluster_size = min_cluster_size
        self.final_clusters = []

    def _calculate_normalized_laplacian(self, affinity_matrix):
        """Calcula a Matriz Laplaciana Normalizada L = I - D^(-1/2) * A * D^(-1/2)."""
        # Calcula a matriz de graus D
        degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
        
        # Para evitar divisão por zero, adiciona um pequeno epsilon onde o grau é 0
        degree_matrix_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
        degree_matrix_inv_sqrt[np.isinf(degree_matrix_inv_sqrt)] = 0
        
        # Calcula a Laplaciana Normalizada
        laplacian_matrix = np.eye(affinity_matrix.shape[0]) - degree_matrix_inv_sqrt @ affinity_matrix @ degree_matrix_inv_sqrt
        return laplacian_matrix

    def _recursive_divide(self, indices, affinity_matrix):
        """
        A função recursiva principal que divide os clusters.
        """
        # Caso base: se o cluster atual for muito pequeno, não o divida.
        if len(indices) < self.min_cluster_size:
            self.final_clusters.append(indices)
            return

        # Extrai a submatriz de afinidade para o cluster atual
        current_affinity = affinity_matrix[np.ix_(indices, indices)]
        
        # Calcula a Laplaciana Normalizada para o cluster atual
        laplacian = self._calculate_normalized_laplacian(current_affinity)
        
        # Calcula os dois menores autovalores e autovetores
        # 'SM' significa 'Smallest Magnitude'
        try:
            # Usamos eigs para eficiência em matrizes esparsas, mas funciona aqui também
            eigenvalues, eigenvectors = eigs(laplacian, k=2, which='SM')
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
        except:
            # Se o cálculo falhar (matriz muito pequena/instável), não divide
            self.final_clusters.append(indices)
            return
            
        # O segundo menor autovalor (índice 1, pois eles vêm ordenados)
        lambda2 = eigenvalues[1]
        
        # Caso base: se o cluster for coeso (λ₂ > threshold), não divide
        if lambda2 > self.lambda_threshold:
            self.final_clusters.append(indices)
            return
        
        # Se chegamos aqui, dividimos o cluster
        # Usa o segundo autovetor (Vetor de Fiedler) para a divisão
        fiedler_vector = eigenvectors[:, 1]
        
        # Divide os índices com base no sinal do vetor de Fiedler
        cluster1_indices = [idx for i, idx in enumerate(indices) if fiedler_vector[i] >= 0]
        cluster2_indices = [idx for i, idx in enumerate(indices) if fiedler_vector[i] < 0]
        
        # Chama a função recursivamente para os dois novos sub-clusters
        if cluster1_indices and cluster2_indices:
            self._recursive_divide(cluster1_indices, affinity_matrix)
            self._recursive_divide(cluster2_indices, affinity_matrix)
        else:
            # Se a divisão não produziu dois clusters válidos, para.
            self.final_clusters.append(indices)

    def fit_predict(self, similarity_matrix):
        """
        Executa o clustering na matriz de similaridade fornecida.

        Args:
            similarity_matrix (np.ndarray): Matriz NxN onde o elemento (i, j) é a 
                                            similaridade entre o documento i e j.
        
        Returns:
            np.ndarray: Um array de rótulos de cluster para cada documento.
        """
        self.final_clusters = []
        initial_indices = list(range(similarity_matrix.shape[0]))
        
        # Inicia o processo de divisão recursiva
        self._recursive_divide(initial_indices, similarity_matrix)
        
        # Converte a lista de clusters (índices) para um array de rótulos
        n_samples = similarity_matrix.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        for i, cluster in enumerate(self.final_clusters):
            labels[cluster] = i
            
        return labels