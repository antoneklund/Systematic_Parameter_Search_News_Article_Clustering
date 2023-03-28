from asyncio.log import logger
import sys
import os
from numpy.core.fromnumeric import mean

import scipy

sys.path.append(os.getcwd())  # noqa

import logging
import pandas as pd
import numpy as np
from sklearn.cluster import k_means, AffinityPropagation
from sklearn import metrics
from dimensionality_reduction.dim_red import reduce_pca, reduce_umap, reduce_tsne

import matplotlib.pyplot as plt
import seaborn as sns
from hdbscan import HDBSCAN, all_points_membership_vectors
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary


LOGGER = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DIMENSIONS = [2, 3, 5, 7, 10, 15, 25, 50]  # , 100, 150, 200, 300

PCA_SETTINGS = [0]

SNACK_UMAP_SETTINGS = [5, 20, 80, 320, 1280, 2560]
AG_UMAP_SETTINGS = [5, 20, 80, 320, 1280, 5120]

SNACK_HDBSCAN_SETTINGS = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
AG_HDBSCAN_SETTINGS = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]

SNACK_KMEANS_SETTINGS = [6, 12, 24, 48, 96, 192, 384]
AG_KMEANS_SETTINGS = [4, 8, 16, 32, 64, 128, 256]
R_KMEANS_SETTINGS = [7, 14, 28, 56, 112, 224, 448]

SNACK_SETTINGS = [SNACK_UMAP_SETTINGS, SNACK_KMEANS_SETTINGS, SNACK_HDBSCAN_SETTINGS]
AG_SETTINGS = [SNACK_UMAP_SETTINGS, AG_KMEANS_SETTINGS, AG_HDBSCAN_SETTINGS]
R_SETTINGS = [SNACK_UMAP_SETTINGS, R_KMEANS_SETTINGS, SNACK_HDBSCAN_SETTINGS]

REPETITIONS = 3

TOPK = 10

LOGGER = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def clustering_pipeline(
    vectors_path,
    dim=2,
    dim_setting=5,
    dataset_settings=SNACK_SETTINGS,
    normalize_before_clustering=True,
    normalize_before_reduction=False,
    dim_red_technique="umap",
):
    time_dim_red = 0
    if dim_red_technique != "raw":
        corpus_df = pd.read_pickle(vectors_path)
        # corpus_df = corpus_df.sample(2000)
        list_of_texts = [text.split() for text in corpus_df["text"].to_list()]
        dictionary = Dictionary(list_of_texts)
        # dictionary.filter_extremes(no_below=5, no_above=1, keep_n=50000)
        eval_tools = [list_of_texts, dictionary]
        vectorization = set_vectorization_string(corpus_df["doc_vec"].to_list())

        vectors = corpus_df["doc_vec"].to_list()

        if normalize_before_reduction:
            LOGGER.info("Normalize before dimensionality reduction.")
            vectors = normalize_l2(vectors)

        LOGGER.info("Applying dimensionality reduction")
        start_0 = time.time()
        if dim_red_technique == "umap":
            vectors = reduce_umap(vectors, dim, dim_setting)
        elif dim_red_technique == "pca":
            vectors = reduce_pca(vectors, dim)
        elif dim_red_technique == "tsne":
            vectors = reduce_tsne(vectors, dim, dim_setting)
        end_0 = time.time()
        time_dim_red = end_0 - start_0

    #############################
    # This part is left from when dimensionality reduction was done directly with Doc2Vec
    else:
        LOGGER.info("Loading vectors from " + vectors_path)
        corpus_df = pd.read_pickle(vectors_path)
        vectors = corpus_df["doc_vec"].to_list()
        vectors = np.array(vectors)
    #############################

    if normalize_before_clustering:
        LOGGER.info("Normalize before clustering.")
        vectors = normalize_l2(vectors)

    scores_df = clustering_loop(
        vectorization,
        vectors,
        corpus_df,
        dim_red_technique,
        dim,
        dim_setting,
        time_dim_red,
        dataset_settings,
        eval_tools
    )
    return scores_df


def clustering_loop(
    vectorization,
    vectors,
    corpus_df,
    dim_red_technique,
    dim,
    dim_setting,
    time_dim,
    dataset_settings,
    eval_tools
):
    kmeans_settings = dataset_settings[1]
    hdbscan_settings = dataset_settings[2]

    # Cluster loop
    scores_df = pd.DataFrame(
        columns=[
            "vectorization",
            "dimensionality_reduction",
            "clustering",
            "dim",
            "dim_setting",
            "clustering_setting",
            "rand_score",
            "mutual_score",
            "topic_coherence",
            "topic_diversity",
            "nr_clusters",
            "time_dim_red",
            "time_clustering",
        ]
    )
    for clustering_technique in ["kmeans", "hdbscan"]:
        LOGGER.info(
            "Clustering with "
            + clustering_technique
            + ". Dimension settings:"
            + str(dim)
            + "D "
            + dim_red_technique
            + " setting "
            + str(dim_setting)
        )
        true_labels = corpus_df["label"].to_list()

        if clustering_technique == "kmeans":
            for setting in kmeans_settings:
                start_1 = time.time()
                _, pred_labels, _ = k_means(vectors, setting, n_init=10)
                end_1 = time.time()
                time_clustering = 0
                time_clustering = end_1 - start_1
                corpus_df["pred_labels"] = pred_labels
                evaluation_scores = evaluate_clusters(vectors, pred_labels, true_labels, corpus_df, eval_tools)
                scores_df = add_evaluation_scores(
                    evaluation_scores,
                    scores_df,
                    vectorization,
                    dim_red_technique,
                    clustering_technique,
                    dim,
                    dim_setting,
                    setting,
                    time_dim,
                    time_clustering,
                )

        elif clustering_technique == "hdbscan":
            for setting in hdbscan_settings:
                start_1 = time.time()
                soft_clustering = True
                if soft_clustering:
                    clusters = HDBSCAN(
                        min_cluster_size=setting,
                        prediction_data=True,
                        core_dist_n_jobs=-1,
                        min_samples=int(np.minimum(setting, 480))
                    ).fit(vectors)
                    pred_labels = [
                        np.argmax(x) for x in all_points_membership_vectors(clusters)
                    ]

                else:
                    clusters = HDBSCAN(
                        min_cluster_size=setting, core_dist_n_jobs=-1
                    ).fit(vectors)
                    pred_labels = clusters.labels_
                end_1 = time.time()
                time_clustering = 0
                time_clustering = end_1 - start_1
                corpus_df["pred_labels"] = pred_labels
                evaluation_scores = evaluate_clusters(vectors, pred_labels, true_labels, corpus_df, eval_tools)
                scores_df = add_evaluation_scores(
                    evaluation_scores,
                    scores_df,
                    vectorization,
                    dim_red_technique,
                    clustering_technique,
                    dim,
                    dim_setting,
                    setting,
                    time_dim,
                    time_clustering,
                )

        else:
            start_1 = time.time()
            affinity_prop = AffinityPropagation().fit(vectors)
            end_1 = time.time()
            pred_labels = affinity_prop.labels_
            time_clustering = 0
            time_clustering = end_1 - start_1
            corpus_df["pred_labels"] = pred_labels
            evaluation_scores = evaluate_clusters(vectors, pred_labels, true_labels, corpus_df, eval_tools)
            scores_df = add_evaluation_scores(
                evaluation_scores,
                scores_df,
                vectorization,
                dim_red_technique,
                clustering_technique,
                dim,
                dim_setting,
                0,
                time_dim,
                time_clustering,
            )

        corpus_df["pred_labels"] = pred_labels

        # plot_2D(corpus_df, vectors)
    return scores_df


def evaluate_clusters(vectors, pred_label, true_label, corpus_df, eval_tools):
    LOGGER.info("Evaluating")
    rand_score = metrics.adjusted_rand_score(pred_label, true_label)
    mutual_score = metrics.adjusted_mutual_info_score(pred_label, true_label)
    # if len(set(pred_label)) > 0:
    #     calinski_harabasz = 0
    #     davies_bouldin = 0
    #     silhouette_score = 0
    # else:
    #     calinski_harabasz = metrics.calinski_harabasz_score(vectors, pred_label)
    #     davies_bouldin = metrics.davies_bouldin_score(vectors, pred_label)
    #     silhouette_score = metrics.silhouette_score(vectors, pred_label)
    
    print("Nr clusters: {}".format(len(np.unique(pred_label))))
    list_of_texts = eval_tools[0]
    dictionary = eval_tools[1]
    topics_df = create_topics_df(corpus_df, max_topics=10000)
    topics_df = add_tfidf_keywords_to_topics_df(topics_df)
    topics = topics_df["keywords"].to_list()
    # print(topics)
    corpus = [dictionary.doc2bow(text) for text in list_of_texts]

    logging.disable()
    nr_clusters = len(np.unique(pred_label))  # NUMBER OF CLUSTERS FOUND
    
    coherence_model = CoherenceModel(topics=topics, corpus=corpus, texts=list_of_texts, dictionary=dictionary, model="c_v", topn=TOPK)
    topic_coherence = coherence_model.get_coherence()
    print("Coherence: {}".format(topic_coherence))

    topic_diversity = topic_diversity_calculation(topics, topk=TOPK)
    # print("Diversity: {}".format(topic_diversity))

    logging.disable(logging.DEBUG)
    return [
        rand_score,
        mutual_score,
        topic_coherence,
        topic_diversity,
        nr_clusters,
    ]


def add_evaluation_scores(
    scores,
    scores_df,
    vectorizing_technique,
    dim_red_technique,
    clustering_technique,
    dim,
    dim_setting,
    cl_setting,
    time_dim_red,
    time_clustering,
):

    add_df = pd.DataFrame(
        [
            [
                vectorizing_technique,
                dim_red_technique,
                clustering_technique,
                dim,
                dim_setting,
                cl_setting,
                scores[0],
                scores[1],
                scores[2],
                scores[3],
                scores[4],
                time_dim_red,
                time_clustering,
            ]
        ],
        columns=[
            "vectorization",
            "dimensionality_reduction",
            "clustering",
            "dim",
            "dim_setting",
            "clustering_setting",
            "rand_score",
            "mutual_score",
            "topic_coherence",
            "topic_diversity",
            "nr_clusters",
            "time_dim_red",
            "time_clustering",
        ],
    )
    scores_df = pd.concat([scores_df, add_df], ignore_index=True)
    return scores_df


def create_topics_df(df, max_topics=10000):
    topics = df["pred_labels"].unique()
    topics = topics[:int(np.minimum(len(topics), max_topics))]

    df = df[df["pred_labels"].isin(topics)]

    topics_df = pd.DataFrame()
    topics_df["topics"] = topics
    topics_df.sort_values("topics")
    text_list = []
    for topic in topics:
        texts = df["text"].loc[df["pred_labels"] == topic]
        all_text = ""
        for text in texts:
            all_text = all_text + text
        text_list.append(all_text)
    topics_df["text"] = text_list
    return topics_df


def add_tfidf_keywords_to_topics_df(df):
    texts = df["text"].to_list()
    extended_stopwords = ["said", "say", "mr", "would", "ve", "just", "yes", "no", "don"]
    stop_words = set(list(ENGLISH_STOP_WORDS))
    stop_words = list(stop_words.union(extended_stopwords))
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts).toarray()
    vocab = tfidf_vectorizer.vocabulary_
    reverse_vocab = {v: k for k, v in vocab.items()}
    idx = tfidf_matrix.argsort(axis=1)
    tfidf_top_n_words = idx[:, -TOPK:]
    top_keywords = [[reverse_vocab.get(item) for item in row] for row in tfidf_top_n_words]
    df["keywords"] = top_keywords
    return df


def normalize_l2(vectors):
    # Center vectors around origo
    mean_vec = np.zeros(len(vectors[0]))
    for v in vectors:
        mean_vec = mean_vec + v
    mean_vec = mean_vec / len(vectors)
    vectors = vectors - mean_vec
    # divide by
    vectors = [v / np.linalg.norm(v) for v in vectors]
    return vectors


def set_vectorization_string(vectors):
    print(vectors[0].shape)
    if vectors[0].shape[0] == 300:
        vectorization = "doc2vec"
    elif (vectors[0].shape[0] == 512) or (vectors[0].shape[0] == 768):
        vectorization = "bert"
    else:
        vectorization = "unknown"
        raise Warning(
            "doc2vec or bert vectors not specified. Vector shape not 300 or 512 or 768."
        )
    return vectorization


def topic_diversity_calculation(topics, topk):
    unique_words = set()
    for topic in topics:
        unique_words = unique_words.union(set(topic))
    td = len(unique_words) / (topk * len(topics))
    return td


def run_all_cluster_systems(
    vectors_path="dim_vectors/doc2vec_300D.pkl",
    save_path="scores/score_data.pkl",
    dataset_settings="snack",
    dim_red_technique="umap",
    start_dim=2,
    normalize_before_clustering=True,
    normalize_before_reduction=False,
):

    dimensions = DIMENSIONS
    dimensions.reverse()
    new_dimensions = []
    for dim in dimensions:
        if dim == start_dim:
            new_dimensions.append(dim)
            new_dimensions.reverse()
            dimensions = new_dimensions
            break
        else:
            new_dimensions.append(dim)

    if dataset_settings is None:
        settings_collection = SNACK_SETTINGS
    elif dataset_settings == "snack":
        settings_collection = SNACK_SETTINGS
    elif dataset_settings == "ag":
        settings_collection = AG_SETTINGS
    elif dataset_settings == "r":
        settings_collection = R_SETTINGS
    else:
        raise Exception("Setting not found. Try 'snack', 'r' or 'ag'.")

    red_settings = settings_collection[0]
    scores_df = pd.DataFrame(
        columns=[
            "vectorization",
            "dimensionality_reduction",
            "clustering",
            "dim",
            "dim_setting",
            "clustering_setting",
            "rand_score",
            "mutual_score",
            "topic_coherence",
            "topic_diversity",
            "nr_clusters",
            "time_dim_red",
            "time_clustering",
        ]
    )

    if dim_red_technique == "pca" or dim_red_technique == "raw":
        red_settings = PCA_SETTINGS
    elif dim_red_technique == "umap":
        red_settings = settings_collection[0]
    else:
        raise Exception("Need to select 'umap' or 'pca'.")

    total_runs = len(dimensions) * REPETITIONS * len(red_settings)
    run_nr = 0
    LOGGER.info("Starting clustering loop with total {} settings.".format(total_runs))
    LOGGER.info("Dataset: " + dataset_settings)
    LOGGER.info("Dimensions: " + str(dimensions))
    LOGGER.info("Reducer settings: " + str(red_settings))
    LOGGER.info("K-Means settings: " + str(settings_collection[1]))
    LOGGER.info("HDBSCAN settings: " + str(settings_collection[2]))

    for dim in dimensions:
        for _ in range(REPETITIONS):
            for dim_setting in red_settings:
                run_nr += 1
                print("Run: (" + str(run_nr) + "/" + str(total_runs) + ")")
                add_df = clustering_pipeline(
                    vectors_path,
                    dim=dim,
                    dim_red_technique=dim_red_technique,
                    dim_setting=dim_setting,
                    dataset_settings=settings_collection,
                    normalize_before_clustering=normalize_before_clustering,
                    normalize_before_reduction=normalize_before_reduction,
                )
                LOGGER.info("Saving results")
                scores_df = pd.concat([scores_df, add_df], ignore_index=True)
                scores_df.to_pickle(save_path)
