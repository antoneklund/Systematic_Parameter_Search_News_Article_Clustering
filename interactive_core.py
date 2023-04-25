import sys
import os

sys.path.append(os.getcwd())  # noqa

import plotly.express as px
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.cluster import k_means
from dimensionality_reduction.dim_red import reduce_umap, reduce_pca
import re

# from dimensionality_reduction.plotting import (
#     split_into_x_and_y_axes,
#     split_into_x_and_y_and_z_axes,
#     convert_dataframe_column_from_float_to_category,
# )
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from stopwordsiso import stopwords as stopwords_iso

import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

SWEDISH_STOP_WORDS = stopwords.words("swedish")
RUSSIAN_STOP_WORDS = stopwords.words("russian")
CHINESE_STOP_WORDS = stopwords_iso(["zh"])
PERSIAN_STOP_WORDS = stopwords_iso(["fa"])

EXTENDED_STOPWORDS = [
    "said",
    "say",
    "mr",
    "would",
    "ve",
    "just",
    "yes",
    "no",
    "don",
    "rowspan",
    "small",
    "bgcolor",
    "center",
    "text",
    "background",
    "flagicon",
    "left",
    "style",
    "align",
    "year",
    "years",
    "scope",
    "class",
    "wikitable",
    "row",
    "f0f0f0",
    "colspan",
    "new",
    "university",
    "school",
    "born",
    "cffcff",
    "affeee" "width",
    "font",
    "98fb98",
    "ffa07a",
    "nbsp",
    "e0e0e0",
    "d5d5d5",
    "ffdead",
    "format",
    "crankdat",
    "loss",
    "size",
    "loss",
    "title",
    "white",
    "yes2",
    "col",
    "ffffff",
    "itf",
    "f0f8ff",
    "rd",
    "nd",
    "st",
    "afeeee",
    "cc9966",
    "fick",
    "bland",
    "första",
    "även",
    "the",
    "död",
    "född",
    "år",
    "the",
    "of",
    "flaggbild",
    "annat",
    "andra",
    "tillsammans",
    "gjorde",
    "border",
    "none",
    "också",
    "samt",
    "in",
    "senare",
    "solid",
    "talet",
    "genom",
    "flera",
    "två",
    "kom",
    "sitt",
    "etta",
    "tog",
    "plats",
    "th",
    "lang",
    "film",
    "valign",
    "ja",
    "right",
    "width",
    "link",
    "win",
    "integer",
    "color",
    "boxstyle_",
    "category",
    "dts",
    "moccasin",
    "bwfic",
    "bwfis",
    "color",
    "zh",
    "middle",
    "fb",
    "tw",
    "gold",
    "percentage",
    "silver",
    "isbn",
    "le",
    "tsl",
    "collapse",
    "nonp",
    "em",
    "px",
    'unsortable',
    'der',
    'bwffs',
    "date",
    "tabel",
    "sorting",
    "ymd",
    "年",
    "月",
    "日",
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
    "人",
    "上",
    "下",
    "的",
    "了",
    "个",
    "等",
    "及",
    "和",
    "在",
    "有",
    "该",
    "为",
    "於",
    "最",
    "更",
    "年月日",
    "中",
    "却",
    "年于",
    "tooltip",
    "round",
    "sortable",
    "value",
]


def limit_hover_string(text):
    return text[:80]


def hdbscan_clustering(vectors, min_cluster_size):
    clusters = HDBSCAN(
        min_cluster_size=min_cluster_size,
        prediction_data=True,
        core_dist_n_jobs=-1,
        # min_samples=100
    ).fit(vectors)
    membership_to_all_clusters = all_points_membership_vectors(clusters)
    pred_labels = [np.argmax(x) for x in membership_to_all_clusters]
    return pred_labels, membership_to_all_clusters


def create_interactive_df_for_2D(df, vectors, pred_labels):
    x_axis, y_axis = split_into_x_and_y_axes(vectors)
    interactive_df = pd.DataFrame()
    interactive_df["text"] = df["text"]
    interactive_df["title"] = df["title"]
    interactive_df["keywords"] = df["keywords"]
    interactive_df["x"] = x_axis
    interactive_df["y"] = y_axis
    interactive_df["label"] = pred_labels
    interactive_df = convert_dataframe_column_from_float_to_category(
        interactive_df, "label"
    )
    return interactive_df


def interactive_figure_in_2D(df):
    doc_vec = df["doc_vec"].to_list()
    vectors = reduce_umap(doc_vec, 2, 20)
    pred_labels = hdbscan_clustering(vectors)
    interactive_df = create_interactive_df_for_2D(df, vectors, pred_labels)

    fig = px.scatter(interactive_df, x="x", y="y", hover_data=["title"], color="label")
    fig.show()


def interactive_figure_in_3D(
    df,
    n_components=3,
    min_cluster_size=5,
    k_means_k=100,
    n_neighbors=10,
    dimension_reduction_method="umap",
    cluster_method="hdbscan",
):
    doc_vec = df["doc_vec"].to_list()
    LOGGER.info("Running dimensionality reduction.")
    if dimension_reduction_method == "umap":
        vectors = reduce_umap(
            doc_vec, n_components_=n_components, n_neighbors_=n_neighbors
        )
    elif dimension_reduction_method == "pca":
        vectors = reduce_pca(doc_vec, n_components)
    else:
        raise (Exception("Not pca or umap."))
    LOGGER.info("Running clustering.")
    if cluster_method == "hdbscan":
        pred_labels, cluster_memberships = hdbscan_clustering(
            vectors, min_cluster_size=min_cluster_size
        )
    elif cluster_method == "kmeans":
        _, pred_labels, _ = k_means(vectors, k_means_k)
        cluster_memberships = []

    LOGGER.info("Creating 3D plot.")
    # vectors = reduce_umap(vectors, n_components_=3, n_neighbors_=n_neighbors)

    interactive_df = create_interactive_df_for_3D(
        df, vectors, pred_labels, cluster_memberships
    )

    fig = px.scatter_3d(
        interactive_df,
        x="x",
        y="y",
        z="z",
        hover_data=["title", "keywords"],
        color="label",
        height=None,
        width=None,
    )
    fig.update_traces(
        marker=dict(size=4, line=dict(width=0.1, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.write_html("testing.html")
    fig.show()
    return interactive_df


def create_interactive_df_for_3D(df, vectors, pred_labels, cluster_memberships):
    x_axis, y_axis, z_axis = split_into_x_and_y_and_z_axes(vectors)
    interactive_df = pd.DataFrame()
    interactive_df["text"] = df["text"]
    interactive_df["title"] = df["title"]
    interactive_df["keywords"] = df["keywords"]
    interactive_df["x"] = x_axis
    interactive_df["y"] = y_axis
    interactive_df["z"] = z_axis
    interactive_df["label"] = pred_labels
    # interactive_df["memberships"] = cluster_memberships.tolist()
    interactive_df = convert_dataframe_column_from_float_to_category(
        interactive_df, "label"
    )
    return interactive_df


def add_tfidf_keywords_to_df(df):
    texts = df["text"].to_list()
    texts = [
        re.sub("[0-9a-f]{6}", "", text) for text in texts
    ]  # remove all hexcolors (f9f9f9, ...)
    texts = [re.sub(r"[0-9]", "", text) for text in texts]  # remove all numbers
    stop_words = create_list_of_stopwords()
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts).toarray()
    vocab = tfidf_vectorizer.vocabulary_
    reverse_vocab = {v: k for k, v in vocab.items()}
    idx = tfidf_matrix.argsort(axis=1)
    tfidf_max20 = idx[:, -10:]
    top_keywords = [[reverse_vocab.get(item) for item in row] for row in tfidf_max20]
    df["keywords"] = top_keywords
    return df


def create_list_of_stopwords():
    extended_stopwords = set(EXTENDED_STOPWORDS)
    swedish_stopwords = set(SWEDISH_STOP_WORDS)
    english_stopwords = set(ENGLISH_STOP_WORDS)
    russian_stopwords = set(RUSSIAN_STOP_WORDS)
    chinese_stopwords = set(CHINESE_STOP_WORDS)
    persian_stopwords = set(PERSIAN_STOP_WORDS)
    stop_words = extended_stopwords.union(swedish_stopwords)
    stop_words = stop_words.union(english_stopwords)
    stop_words = stop_words.union(russian_stopwords)
    stop_words = stop_words.union(chinese_stopwords)
    stop_words = stop_words.union(persian_stopwords)
    stop_words = list(stop_words)
    return stop_words


def create_topics_df(df):
    topics = df["label"].unique()
    topics_df = pd.DataFrame()
    topics_df["topic_id"] = topics
    topics_df.sort_values("topic_id")
    text_list = []
    for topic in topics:
        texts = df["text"].loc[df["label"] == topic]
        all_text = ""
        for text in texts:
            all_text = all_text + text
        text_list.append(all_text)
    topics_df["text"] = text_list
    return topics_df


def select_articles_from_topic(df, topic_nr):
    return df[df["label"] == topic_nr]


def select_topic_articles_from_article_id(df, article_id):
    topic_nr = df["label"].iloc[article_id]
    return select_articles_from_topic(df, topic_nr)


# def main():


#     interactive_df = interactive_figure_in_3D(df)


# if __name__ == "__main__":
#     main()
