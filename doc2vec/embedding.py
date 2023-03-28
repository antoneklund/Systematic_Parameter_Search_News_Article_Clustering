"""Doc2Vec model loading and creation functions.

This module expects input corpus data in the form of a Pandas
DataFrame. It is also able to build on top of said DataFrame. See the
'load_corpus_dataframe' function inside 'utilities.py'.
"""
import logging
import os.path
import sys
from typing import Generator

import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
from pandas import DataFrame

LOGGER = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

N_DIMENSIONS = 300
TRAINING_EPOCHS = 15
N_WORKERS = 4


def tokenized_tagged_corpus(corpus_df: DataFrame) -> Generator[TaggedDocument, None, None]:
    """Tokenize given text data and yield TaggedDocuments with id tag.

    The given Pandas DataFrame should have (at least) these columns of
    data: ['article_id', 'text']

    TaggedDocument objects is the input fromat required by the Gensim
    Doc2Vec model.

    Arguments:
        corpus_df: Dataframe of texts to tokenize and tag. Should
            contain, at least, the following columns: [article_id, text]
    """
    for _, article in corpus_df.iterrows():
        tokenized_text = word_tokenize(article['text'])

        yield TaggedDocument(tokenized_text[0:512], [article['article_id']])


def load_model(filename: str) -> Doc2Vec:
    """Load and return model at given path."""
    if os.path.isfile(filename):
        LOGGER.info("Loading model doc2vec model '%s'", filename)
        model = gensim.models.doc2vec.Doc2Vec.load(filename)
    else:
        raise FileNotFoundError("Model not found.")

    return model


def train_model(corpus_df: DataFrame,
                filename: str = None,
                vector_size: int = N_DIMENSIONS,
                epochs: int = TRAINING_EPOCHS,
                n_workers: int = N_WORKERS
                ) -> Doc2Vec:
    """Create, train and return given Gensim Doc2Vec model.

    Arguments:
        corpus_df: DataFrame containing corpus data. Should contain,
                   at least, the following columns:
                   ['article_id', 'text']
        filename: if set, will save model at this location
        vector_size: number of dimensions of resulting embeddings
        epochs: number of epochs to train model for
        n_workers: number of workers to parallelize training with
    """
    LOGGER.info("Creating new model with vector size %s", vector_size)

    LOGGER.info("Tokenizing input data")
    train_corpus = list(tokenized_tagged_corpus(corpus_df))

    LOGGER.info("Building model")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size,
                                          min_count=2,
                                          workers=n_workers,
                                          dm=1,)

    model.build_vocab(train_corpus)

    LOGGER.info("Training model for %s epochs", epochs)
    model.train(train_corpus,
                total_examples=model.corpus_count,
                epochs=epochs)
    LOGGER.info("Completed training",)

    if filename:
        model.save(filename)
        LOGGER.info("Doc2vec model '%s' saved", filename)

    return model


def populate_with_embeddings(corpus_df: DataFrame, model: Doc2Vec) -> None:
    """Populate dataframe with vector embeddings from given model."""
    if 'doc_vec' not in corpus_df.columns:
        corpus_df['doc_vec'] = None

    for _, article in corpus_df.iterrows():
        article_id = article['article_id']

        doc_vec = model.docvecs[article_id]

        df_row = corpus_df.loc[corpus_df['article_id'] == article_id].index[0]

        # print(doc_vec)
        corpus_df.at[df_row, 'doc_vec'] = doc_vec

    # doc_vecs = corpus_df["old_doc_vec"].to_list()
    # print(doc_vecs)
    # norms = [np.linalg.norm(v) for v in doc_vecs]
    # max_norm = np.max(norms)
    # print(max_norm)
    # corpus_df["doc_vec"] = corpus_df["old_doc_vec"]/max_norm
