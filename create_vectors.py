"""Argument parser for running dataset cleaning and creating vectors.
It is recommended to always give a custom corpus_path and save_path
to not overwrite anything.

Examples:
Clean and prepare all datasets:
    python create_vectors.py -p

Create Doc2Vec vectors:
    python create_vectors.py -o d2v -c datasets/snack_doc2vec.pkl
        -s dim_vectors/snack_doc2vec_vectors.pkl

Train a BERT model (requires GPU):
    python create_vectors.py -t -c datasets/snack_bert.pkl -m models/bert.torch

Run inference on the BERT vectors with a BERT model:
    python create_vectors.py -o bert -c datasets/snack_bert.pkl -m models/bert.torch
        -s dim_vectors/snack_bert_vectors.pkl

Alternatively, use small BERT to run inference without training:
    python create_vectors.py -o small_bert -c datasets/snack_bert.pkl
        -s dim_vectors/snack_small_bert_vectors.pkl

"""
import pandas as pd
from dimensionality_reduction.create_embeddings import (
    create_doc2vec_vectors,
    prepare_all_datasets,
    snack_corpus
)
from bert.inference_torch import trained_bert_inference, small_bert_inference, inference_bert_uncased
from bert.train_torch import train_torch_model
from argparse import ArgumentParser


ARG_PARSER = ArgumentParser(
    description=("Create vectors ready for the clustering loop.")
)

ARG_PARSER.add_argument(
    "-c",
    "--corpus_path",
    default="datasets/snack_doc2vec.pkl",
    help=(
        "Path to the data. Must be a DataFrame in the form of a .pkl with"
        "the columns 'text' and 'label'."
    ),
)

ARG_PARSER.add_argument(
    "-s",
    "--save_path",
    default="dim_vectors/vectors.pkl",
    help=("Where to save the scores after completed clustering runs."),
)

ARG_PARSER.add_argument(
    "-m",
    "--model_path",
    default="models/bert.torch",
    help=("Path to torch BERT model." "Can be created with train_torch.py"),
)

ARG_PARSER.add_argument(
    "-o",
    "--option",
    default="d2v",
    help=(
        "Choose which vectorization technique to use."
        "'all' will prepare all datasets from scratch."
        "'d2v' for Doc2Vec, 'bert' for trained BERT"
        "and 'small_bert' for small BERT."
    ),
)

ARG_PARSER.add_argument(
    "-d",
    "--dataset",
    default="snack",
    help=(
        "Which dataset it is: [snack, reuters, ag]"
    ),
)

ARG_PARSER.add_argument(
    "-p",
    "--prepare",
    action="store_true",
    help=(
        "If this flag is true, prepare all datasets"
        " for doc2vec and bert vector creation."
    ),
)

ARG_PARSER.add_argument(
    "-po",
    "--prepare_one",
    action="store_true",
    help=(
        "If this flag is true, prepare one specified dataset"
        " for doc2vec and bert vector creation. Spcify with -c"
    ),
)

ARG_PARSER.add_argument(
    "-t",
    "--train_bert",
    action="store_true",
    help=(
        "If this flag is true, train bert model."
        "model_path will be the save_path for the BERT model."
        "corpus_path should be to a BERT-prepared dataset."
    ),
)


def main():
    args = ARG_PARSER.parse_args()

    if args.prepare:
        prepare_all_datasets()
        return

    elif args.prepare_one:
        snack_corpus(raw_path=args.corpus_path, dataset_name=args.dataset)

    elif args.train_bert:
        train_torch_model(args.corpus_path, args.model_path)

    elif args.option == "d2v":
        create_doc2vec_vectors(corpus_path=args.corpus_path, save_path=args.save_path)

    elif args.option == "bert":
        corpus_df = pd.read_pickle(args.corpus_path)
        inference_bert_uncased(corpus_df, args.model_path, args.save_path)

    elif args.option == "small_bert":
        corpus_df = pd.read_pickle(args.corpus_path)
        small_bert_inference(corpus_df, args.save_path)
   
    elif args.option == "bert-base":
        corpus_df = pd.read_pickle(args.corpus_path)
        if args.dataset == "snack":
            corpus_df["text"] = corpus_df["text"]
        if args.dataset == "reuters":
            corpus_df["text"] = corpus_df["story_text"]
        inference_bert_uncased(corpus_df, args.save_path)    

    else:
        raise Exception("option must be [d2v, bert, small_bert]")


if __name__ == "__main__":
    main()
