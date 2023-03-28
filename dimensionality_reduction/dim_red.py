from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
TODO: Doc2Vec, BERT, BERTopic, Import articles and add pre-processing  
    pre-process 
    load vectors
    load labels
    for dim in dims
        perform dimensionality reduction
        clusterability measurements
        clustering + extrinsic measurements 
        intrinsic measurements

"""
DIMENSIONS = [2, 3, 5, 7, 10, 15, 25, 50, 100, 150, 200, 300]


def reduce_pca(vecs, n_components):
    pca = PCA(n_components)
    pca_vecs = pca.fit_transform(vecs)
    return pca_vecs


def reduce_umap(vecs, n_components_=3, n_neighbors_=5):
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors_, n_components=n_components_)
    umap_vecs = umap_reducer.fit_transform(vecs)
    return umap_vecs


def reduce_tsne(vecs, n_components_=3, perplexity_=30):
    tsne_vecs = TSNE(n_components=n_components_, perplexity=perplexity_).fit_transform(
        vecs
    )
    return tsne_vecs


def plot_dim_reduction(vecs, reduction_method="", labels=None):
    if labels is None:
        plt.scatter(vecs[:, 0], vecs[:, 1], s=0.1)
        plt.title("Plot of reduced vector space in 2D with %s".format(reduction_method))
        plt.show()


def take_first(obj):
    return obj[0]


def main():
    for dim in DIMENSIONS:
        corpus = pd.read_pickle("dim_vectors/doc2vec_{}D.pkl".format(dim))
        vecs = corpus["doc_vec"].to_list()
        pca_vecs, pca = reduce_pca(vecs, dim)
        print(pca.explained_variance_ratio_)
        plt.plot(range(1, dim + 1), pca.explained_variance_ratio_)
        plt.scatter(range(1, dim + 1), pca.explained_variance_ratio_, c="red")
        plt.xticks(range(1, dim + 1, 5))
        plt.title("Explained variance for {} principle components".format(dim))
        plt.ylabel("Explained variance")
        plt.xlabel("PCA component")
        plt.show()


if __name__ == "__main__":
    main()
