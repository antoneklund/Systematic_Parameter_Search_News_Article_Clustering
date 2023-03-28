'''Argument parser for controlling the clustering loop.
The flags -nbc and -nbr should be added everytime when normalizing. Default is without
normalizing.

start-dim 

Example:
Run clustering loop:
    python clustering_loop.py -v dim_vectors/snack_doc2vec_vectors.pkl 
        -s scores/scores_snack_d2v<DATE>.pkl

Run with normalizing before clustering:
    python clustering_loop.py -v dim_vectors/snack_doc2vec_vectors.pkl 
        -s scores/scores_snack_d2v<DATE>.pkl -nbc

Run with normalizing before dimensionality reduction and clustering:
    python clustering_loop.py -v dim_vectors/snack_doc2vec_vectors.pkl 
        -s scores/scores_snack_d2v<DATE>.pkl -nbr -nbc
    

'''

from dimensionality_reduction.cluster import run_all_cluster_systems
from argparse import ArgumentParser


ARG_PARSER = ArgumentParser(
    description=("Run the clustering loop for 1 dataset and for UMAP or PCA.")
)

ARG_PARSER.add_argument(
    "-v",
    "--vectors_path",
    default="dim_vectors/doc2vec_300D.pkl",
    help=(
        "Path to .pkl file with prepared vectors."
        "Examples: [dim_vectors/doc2vec_300D.pkl, dim_vectors/bert_prep_211028.pkl]"
        "The .pkl file must contain the columns 'dim_vector' and 'label' for each article."
    ),
)

ARG_PARSER.add_argument(
    "-s",
    "--save_path",
    default="scores/scores_data.pkl",
    help=("Where to save the scores after completed clustering runs."),
)

ARG_PARSER.add_argument(
    "-d",
    "--dataset_settings",
    default="snack",
    help=("Choose what settings to use with the dataset. [snack, ag, r]"),
)

ARG_PARSER.add_argument(
    "-r",
    "--dim_red_technique",
    default="umap",
    help=("Choose which dimensionality reduction technique. [pca, umap]."),
)

ARG_PARSER.add_argument(
    "-sd",
    "--start_dim",
    type=int,
    default=2,
    help=(
        "Choose which dimension to start the run. Useful when needing to cancel"
        "in the middle of long runs."
    ),
)

ARG_PARSER.add_argument(
    "-nbc",
    "--normalize_before_clustering",
    action="store_true",
    help=("Choose if to normalize before clustering."),
)

ARG_PARSER.add_argument(
    "-nbr",
    "--normalize_before_reduction",
    action="store_true",
    help=("Choose if to normalize before dimensionality reduction."),
)


def main():
    args = ARG_PARSER.parse_args()
    print(args.normalize_before_clustering)
    run_all_cluster_systems(
        vectors_path=args.vectors_path,
        save_path=args.save_path,
        dataset_settings=args.dataset_settings,
        dim_red_technique=args.dim_red_technique,
        start_dim=args.start_dim,
        normalize_before_clustering=args.normalize_before_clustering,
        normalize_before_reduction=args.normalize_before_reduction,
    )


if __name__ == "__main__":
    main()
