from calendar import c
import logging
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.getcwd())  # noqa

from text_preprocessing.nltk_preprocessor import NltkPreprocessor
from text_preprocessing.clean_corpus import make_corpus
from doc2vec.embedding import train_model, populate_with_embeddings
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from pprint import pprint
from text_preprocessing.text_cleaner import clean_text


LOGGER = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

DIMENSIONS = [1, 2, 3, 5, 7, 10, 15, 25, 50, 100, 150, 200, 300]
DIMENSIONS = [300]


def load_20newsgroups():
    newsgroups_train = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )
    pprint(list(newsgroups_train.target_names))
    # print(newsgroups_train.data[0])
    return newsgroups_train.data, newsgroups_train.target


def combine_labels_20newsgroup(corpus_df):
    '''Combines the labels of 20newsgroup dataset since they are
        a bit unbalanced and some of the newsgroups are similar subjects.
    
    '''
    labels = corpus_df["label"].to_list()
    new_labels = []
    for label in labels:
        if label == 1:
            new_labels.append(0)  # atheism
        elif label < 7:
            new_labels.append(1)  # computer
        elif label == 7:
            new_labels.append(2)  # forsale
        elif label < 10:
            new_labels.append(3)  # automobile
        elif label < 12:
            new_labels.append(4)  # sport
        elif label == 14:
            new_labels.append(5)  # medical
        elif label < 16:
            new_labels.append(6)  # science
        elif label == 16:
            new_labels.append(7)  # religion
        elif label < 20:
            new_labels.append(8)  # politics
        else:
            new_labels.append(7)
    assert len(new_labels) == len(labels)
    corpus_df["label"] = new_labels
    # print(corpus_df["label"].describe())
    return corpus_df


def convert_to_dataframe(data, labels):
    '''Takes the text and labels and converts them into a pd.DataFrame.
        Cleans the text in preparation for Doc2Vec.
        Removes Punctuation.
    
    '''
    corpus_dict = []
    for article_id, text in enumerate(data):
        # corpus_df = corpus_df.append(pd.DataFrame({"article_id": [article_id],
        #                                "text": [text]}),
        #                                ignore_index=True)
        corpus_dict.append({"article_id": article_id, "text": text})

    cleaned_corpus = make_corpus(
        raw_data=corpus_dict,
        preprocessor_name="nltk",
        language="english",
        split=False,
        remove_punctuations=True,
        remove_special_and_digits=True,
        remove_stopwords=True,
    )

    corpus_df = pd.DataFrame(cleaned_corpus["corpus"])
    corpus_df["label"] = labels

    return corpus_df


def create_doc2vec_embedding(dim, corpus_df, epochs=15):
    '''Creates dim large doc2vec embeddings from the texts of corpus_df.
    
    '''
    model = train_model(
        corpus_df=corpus_df, vector_size=dim, epochs=epochs
    )
    return model, corpus_df


def huffpost_corpus():
    corpus_df = pd.read_pickle("datasets/huffpost_test_split_15C_raw.pkl")
    # corpus_df["label"] = 0 # REMOVE LATER

    texts = corpus_df["text"].to_list()
    labels = corpus_df["label"].to_list()
    corpus_df = convert_to_dataframe(texts, labels)
    corpus_df.to_pickle("datasets/huffpost_final_220221.pkl")

    bert_df = pd.DataFrame(columns=["text", "label"])
    bert_df["text"] = texts
    bert_df["label"] = labels
    bert_df = clean_for_bert(bert_df)
    bert_df.to_pickle("datasets/huffpost_bert_220221.pkl")


def reuters_corpus():
    corpus_df = pd.read_pickle("datasets/REUTERS_RAW.pkl")
    corpus_df = corpus_df.drop_duplicates()

    LOGGER.info("Cleaning Reuters specific text.")
    clean_df = clean_reuters(corpus_df)

    LOGGER.info("Clean and create Doc2Vec-ready dataset.")
    texts = clean_df["text"].to_list()
    labels = clean_df["label"].to_list()
    doc2ve_df = convert_to_dataframe(texts, labels)
    doc2ve_df.to_pickle("datasets/reuters_doc2vec.pkl")

    LOGGER.info("Clean and create BERT-ready dataset.")
    bert_df = pd.DataFrame(columns=["text", "label"])
    bert_df["text"] = texts
    bert_df["label"] = labels
    bert_df = clean_for_bert(bert_df)
    bert_df.to_pickle("datasets/reuters_bert.pkl")


def newsgroup_corpus():
    data, labels = load_20newsgroups()
    corpus_df = convert_to_dataframe(data, labels)
    corpus_df = combine_labels_20newsgroup(corpus_df)
    corpus_df.to_pickle("datasets/20newsgroup_doc2vec.pkl")

    bert_df = pd.DataFrame(columns=["text", "label"])
    bert_df["text"] = data
    bert_df["label"] = labels
    bert_df = clean_for_bert(bert_df)
    bert_df = combine_labels_20newsgroup(bert_df)
    print(bert_df["text"].iloc[1])
    bert_df.to_pickle("datasets/20newsgroup_bert.pkl")


def snack_corpus(raw_path="datasets/SNACK_20k_latest.pkl", dataset_name="snack"):
    LOGGER.info("SNACK dataset")
    corpus_df = pd.read_pickle(raw_path)
    corpus_df = corpus_df[corpus_df["label"] != 6] # Remove the 6th category since it is too small
    texts = corpus_df["text"].to_list()
    labels = corpus_df["label"].to_list()

    LOGGER.info("Clean and create Doc2Vec-ready dataset.")
    corpus_df = convert_to_dataframe(texts, labels)
    corpus_df.to_pickle(f"datasets/{dataset_name}_doc2vec_clean.pkl")

    LOGGER.info("Clean and create BERT-ready dataset.")
    bert_df = pd.DataFrame(columns=["text", "label"])
    bert_df["text"] = texts
    bert_df["label"] = labels
    bert_df = clean_for_bert(bert_df)
    bert_df.to_pickle(f"datasets/{dataset_name}_bert_clean.pkl")


def ag_corpus():
    LOGGER.info("SNACK/AG dataset")
    corpus_df = pd.read_pickle("datasets/AG_NEWS_RAW.pkl")
    texts = corpus_df["text"].to_list()
    labels = corpus_df["label"].to_list()

    LOGGER.info("Clean and create Doc2Vec-ready dataset.")
    corpus_df = convert_to_dataframe(texts, labels)
    corpus_df.to_pickle("datasets/ag_doc2vec.pkl")

    LOGGER.info("Clean and create BERT-ready dataset.")
    bert_df = pd.DataFrame(columns=["text", "label"])
    bert_df["text"] = texts
    bert_df["label"] = labels
    bert_df = clean_for_bert(bert_df)
    bert_df.to_pickle("datasets/ag_bert.pkl")


def clean_for_bert(corpus_df):
    '''Specific function for cleaning text in preparation for BERT.
        Keeps punctuation. Without punctuation, the later BERT functions
        won't work since sentences are splitted there.
    '''
    train_df = pd.DataFrame(columns=["text", "label"])
    train_df["label"] = corpus_df["label"]  # publisher_id for SNACK
    for i, row in corpus_df.iterrows():
        body = row["text"]  # "body for SNACK"
        cleaned_body = clean_text(body, False, True, None, False, True, False)
        train_df["text"].at[i] = cleaned_body
    return train_df


def clean_reuters(corpus_df):
    clean_df = pd.DataFrame(columns=["text", "label"])
    clean_df["label"] = corpus_df["label"]
    for i, row in corpus_df.iterrows():
        body = row["story_text"]
        cleaned_body = clean_text(
            body, False, True, ["Reuters", "reuters", "REUTERS"], False, False, True
        )
        clean_df["text"].at[i] = cleaned_body
    return clean_df


def create_doc2vec_vectors(
    corpus_path="datasets/ag_news_train.pkl",
    save_path="dim_vectors/doc2vec_ag_test.pkl",
    create_many_dimensions=False,
):
    '''Creates vectors for 
    
    '''

    LOGGER.info("Loading data")
    raw_corpus_df = pd.read_pickle(corpus_path)
    data = raw_corpus_df["text"].to_list()  # "body"
    labels = raw_corpus_df["label"].to_list()
    # corpus_df = convert_to_dataframe(data, labels)
    # embeddings = []
    training_df = raw_corpus_df.copy()

    ##################
    # Artifact from when dimensionality reduction was done with doc2vec

    if create_many_dimensions:
        for dim in DIMENSIONS:
            LOGGER.info("Creating model for %iD", dim)
            model, corpus_df = create_doc2vec_embedding(dim, training_df)
            populate_with_embeddings(corpus_df, model)
            LOGGER.info(
                "Saving model at dim_vectors/doc2vec_ag_{}D.pkl".format(str(dim))
            )
            corpus_df.to_pickle("dim_vectors/doc2vec_ag2_{}D.pkl".format(str(dim)))
    ##################

    else:
        LOGGER.info("Creating model for 300D")
        model, corpus_df = create_doc2vec_embedding(300, training_df)
        populate_with_embeddings(corpus_df, model)
        LOGGER.info("Saving model at {}".format(save_path))
        corpus_df.to_pickle(save_path)


def prepare_all_datasets():
    ag_corpus()
    reuters_corpus()
    snack_corpus()


if __name__ == "__main__":
    snack_corpus()
