import re
import sys
import os
import warnings

import numpy as np

sys.path.append(os.getcwd())  # noqa

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dimensionality_reduction.cluster import (
    SNACK_HDBSCAN_SETTINGS,
    SNACK_KMEANS_SETTINGS,
    AG_KMEANS_SETTINGS,
    DIMENSIONS,
    SNACK_SETTINGS,
    AG_SETTINGS,
    R_KMEANS_SETTINGS,
    R_SETTINGS,
    normalize_l2,
)
from dimensionality_reduction.dim_red import reduce_pca, reduce_umap

warnings.simplefilter(action="ignore", category=FutureWarning)

UMAP_SETTING_POSITION = 0
K_MEANS_SETTING_POSITION = 1
HDBSCAN_SETTING_POSITION = 2

px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches


def plot_2D(df, vectors, labels=None, dataset_identifier="", plot_title="", crop=None):
    plot_df = create_plot_dataframe_depending_on_labels(df, labels)
    plot_df = convert_numeric_labels_to_strings(plot_df, dataset_identifier)
    x_axis, y_axis = split_into_x_and_y_axes(vectors)
    plot_df = create_and_add_x_y_columns_to_dataframe(x_axis, y_axis, plot_df)
    if crop != None:
        [x1, x2, y1, y2] = crop
        plot_df = keep_rows_less_than(plot_df, "x", limit=x1, inverse=False)
        plot_df = keep_rows_less_than(plot_df, "x", limit=x2, inverse=True)
        plot_df = keep_rows_less_than(plot_df, "y", limit=y1, inverse=False)
        plot_df = keep_rows_less_than(plot_df, "y", limit=y2, inverse=True)

    print(plot_df)
    plot_scatterplot_with_colored_labels_from_dataframe(plot_df, plot_title)


def create_plot_dataframe_depending_on_labels(df, labels):
    if labels is None:
        labels = df["label"]
    plot_df = pd.DataFrame()
    plot_df["label"] = labels
    plot_df["label_string"] = ""
    return plot_df


def convert_numeric_labels_to_strings(plot_df, dataset_identifier):
    if dataset_identifier == "SNACK":
        plot_df.loc[plot_df["label"] == 0, "label_string"] = "TECHNOLOGY"
        plot_df.loc[plot_df["label"] == 1, "label_string"] = "FOOD & DRINK"
        plot_df.loc[plot_df["label"] == 2, "label_string"] = "SPORTS"
        plot_df.loc[plot_df["label"] == 3, "label_string"] = "STOCKS"
        plot_df.loc[plot_df["label"] == 4, "label_string"] = "CONFLICTS"
        plot_df.loc[plot_df["label"] == 5, "label_string"] = "MOVIES & TV-SERIES"
    elif dataset_identifier == "AG":
        plot_df.loc[plot_df["label"] == 0, "label_string"] = "WORLD"
        plot_df.loc[plot_df["label"] == 1, "label_string"] = "SPORT"
        plot_df.loc[plot_df["label"] == 2, "label_string"] = "BUSINESS"
        plot_df.loc[plot_df["label"] == 3, "label_string"] = "SCIENCE & TECH"
    elif dataset_identifier == "REUTERS":
        plot_df["label_string"] = plot_df["label"]
    else:
        plot_df["label_string"] = plot_df["label"]
    return plot_df


def split_into_x_and_y_axes(vectors):
    X_COLUMN_IN_VECTOR = 0
    Y_COLUMN_IN_VECTOR = 1
    x_axis = [extract_from_vector(vector, X_COLUMN_IN_VECTOR) for vector in vectors]
    y_axis = [extract_from_vector(vector, Y_COLUMN_IN_VECTOR) for vector in vectors]
    return x_axis, y_axis


def split_into_x_and_y_and_z_axes(vectors):
    X_COLUMN_IN_VECTOR = 0
    Y_COLUMN_IN_VECTOR = 1
    Z_COLUMN_IN_VECTOR = 2
    x_axis = [extract_from_vector(vector, X_COLUMN_IN_VECTOR) for vector in vectors]
    y_axis = [extract_from_vector(vector, Y_COLUMN_IN_VECTOR) for vector in vectors]
    z_axis = [extract_from_vector(vector, Z_COLUMN_IN_VECTOR) for vector in vectors]
    return x_axis, y_axis, z_axis


def extract_from_vector(vector, column):
    return vector[column]


def create_and_add_x_y_columns_to_dataframe(x_axis, y_axis, plot_df):
    plot_df["x"] = x_axis
    plot_df["y"] = y_axis
    return plot_df


def keep_rows_less_than(df, axis="x", limit=1, inverse=False):
    if inverse:
        df = df[df[axis] > limit]
    else:
        df = df[df[axis] < limit]
    return df


def plot_scatterplot_with_colored_labels_from_dataframe(plot_df, plot_title):
    fig = plt.figure()
    scatterplot = sns.scatterplot(
        x="x",
        y="y",
        hue="label_string",
        data=plot_df,
        palette=sns.color_palette("hls", len(plot_df["label_string"].unique())),
        alpha=0.5,
        s=5,
        legend="full",
        hue_order=[
            "MARKETS BONDS",
            "SOCCER ENGLAND",
            "NATURAL DISASTER",
            "AUTO",
            "FILM",
            "ENVIRONMENT",
            "USA POLITICS",
        ],
    )
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    plt.title(plot_title)
    scatterplot.set(xticklabels=[])
    scatterplot.set(yticklabels=[])
    scatterplot.set(xlabel=None)
    scatterplot.set(ylabel=None)
    scatterplot.tick_params(bottom=False)
    scatterplot.tick_params(left=False)

    fig.savefig("scatterplot", bbox_inches="tight")
    plt.show()


def calculate_mean_scores_over_clustering_settings(scores_df, settings):
    k_means_settings, hdbscan_settings = extract_k_means_and_hdbscan_settings(settings)

    average_df = pd.DataFrame()
    for dim in DIMENSIONS:
        for cl_setting in k_means_settings:
            mean_df = mean_over_one_clustering_setting(scores_df, dim, cl_setting)
            average_df = average_df.append(mean_df, ignore_index=True)
        for cl_setting in hdbscan_settings:
            mean_df = mean_over_one_clustering_setting(scores_df, dim, cl_setting)
            average_df = average_df.append(mean_df, ignore_index=True)
    average_df = average_df.dropna(how="all")
    print(scores_df)
    return average_df


def mean_over_one_clustering_setting(scores_df, dim, cl_setting):
    mean_df = scores_df[
        (scores_df["dim"] == dim) & (scores_df["clustering_setting"] == cl_setting)
    ].mean()
    return mean_df


def calculate_mean_scores_over_clustering_methods(scores_df, plot_string):
    average_df = pd.DataFrame()
    for dim in DIMENSIONS:
        mean_df = mean_over_one_clustering_method(scores_df, dim, "kmeans", plot_string)
        average_df = average_df.append(mean_df, ignore_index=True)
        mean_df = mean_over_one_clustering_method(
            scores_df, dim, "hdbscan", plot_string
        )
        average_df = average_df.append(mean_df, ignore_index=True)
    average_df = average_df.dropna(how="all")
    return average_df


def mean_over_one_clustering_method(scores_df, dim, clustering_method, plot_string):
    mean_df = scores_df[
        (scores_df["dim"] == dim) & (scores_df["clustering"] == clustering_method)
    ].mean()
    if pd.notna(mean_df["dim"]):
        mean_df["clustering"] = clustering_method
        mean_df["configuration"] = plot_string + "_" + clustering_method
    return mean_df


def calculate_mean_scores_over_dimensions(scores_df):
    average_df = pd.DataFrame()
    for dim in DIMENSIONS:
        mean_df = mean_over_one_dimension(scores_df, dim)
        average_df = average_df.append(mean_df, ignore_index=True)
    average_df = average_df.dropna(how="all")
    return average_df


def mean_over_one_dimension(scores_df, dim):
    mean_df = scores_df[(scores_df["dim"] == dim)].mean()
    return mean_df


def calculate_median_scores_over_clustering_settings(scores_df, settings):
    k_means_settings, hdbscan_settings = extract_k_means_and_hdbscan_settings(settings)

    average_df = pd.DataFrame()
    for dim in DIMENSIONS:
        for cl_setting in k_means_settings:
            median_df = median_over_one_clustering_setting(scores_df, dim, cl_setting)
            average_df = average_df.append(median_df, ignore_index=True)
        for cl_setting in hdbscan_settings:
            median_df = median_over_one_clustering_setting(scores_df, dim, cl_setting)
            average_df = average_df.append(median_df, ignore_index=True)
    average_df = average_df.dropna(how="all")
    print(scores_df)
    return average_df


def median_over_one_clustering_setting(scores_df, dim, cl_setting):
    median_df = scores_df[
        (scores_df["dim"] == dim) & (scores_df["clustering_setting"] == cl_setting)
    ].median()
    return median_df


def calculate_median_scores_over_clustering_methods(scores_df, plot_string):
    average_df = pd.DataFrame()
    for dim in DIMENSIONS:
        median_df = median_over_one_clustering_method(
            scores_df, dim, "kmeans", plot_string
        )
        average_df = average_df.append(median_df, ignore_index=True)
        median_df = median_over_one_clustering_method(
            scores_df, dim, "hdbscan", plot_string
        )
        average_df = average_df.append(median_df, ignore_index=True)
    average_df = average_df.dropna(how="all")
    return average_df


def median_over_one_clustering_method(scores_df, dim, clustering_method, plot_string):
    median_df = scores_df[
        (scores_df["dim"] == dim) & (scores_df["clustering"] == clustering_method)
    ].median()
    if pd.notna(median_df["dim"]):
        median_df["clustering"] = clustering_method
        median_df["configuration"] = plot_string + "_" + clustering_method
    return median_df


def calculate_median_scores_over_dimensions(scores_df):
    average_df = pd.DataFrame()
    for dim in DIMENSIONS:
        median_df = median_over_one_dimension(scores_df, dim)
        average_df = average_df.append(median_df, ignore_index=True)
    average_df = average_df.dropna(how="all")
    return average_df


def median_over_one_dimension(scores_df, dim):
    median_df = scores_df[(scores_df["dim"] == dim)].median()
    return median_df


def calculate_mean_scores_over_all_settings(scores_df, settings):
    k_means_settings, hdbscan_settings = extract_k_means_and_hdbscan_settings(settings)
    vectorization_name, reduction_name = extract_techniques_string_from_dataframe(
        scores_df
    )
    reduction_settings = extract_unique_reduction_settings(scores_df)
    all_mean_df = create_dataframe_with_scores_columns()
    for dim in DIMENSIONS:
        for dim_setting in reduction_settings:
            for cl_setting in k_means_settings:
                mean_df = calculate_mean_over_same_setting_rows(
                    scores_df, dim, dim_setting, cl_setting
                )
                appendable_df = create_appendable_dataframe_with_technique_names(
                    mean_df, vectorization_name, reduction_name, "kmeans"
                )
                all_mean_df = append_to_all_mean_df(appendable_df, all_mean_df)
            for cl_setting in hdbscan_settings:
                mean_df = calculate_mean_over_same_setting_rows(
                    scores_df, dim, dim_setting, cl_setting
                )
                appendable_df = create_appendable_dataframe_with_technique_names(
                    mean_df, vectorization_name, reduction_name, "hdbscan"
                )
                all_mean_df = append_to_all_mean_df(appendable_df, all_mean_df)

    all_mean_df = all_mean_df.dropna()
    all_mean_df = set_floats_to_ints_after_mean_calculations(all_mean_df)
    return all_mean_df


def extract_k_means_and_hdbscan_settings(settings):
    k_means_settings = settings[K_MEANS_SETTING_POSITION]
    hdbscan_settings = settings[HDBSCAN_SETTING_POSITION]
    return k_means_settings, hdbscan_settings


def extract_umap_and_k_means_and_hdbscan_settings(settings):
    umap_settings = settings[UMAP_SETTING_POSITION]
    k_means_settings = settings[K_MEANS_SETTING_POSITION]
    hdbscan_settings = settings[HDBSCAN_SETTING_POSITION]
    return umap_settings, k_means_settings, hdbscan_settings


def extract_techniques_string_from_dataframe(scores_df):
    vectorization_technique = scores_df["vectorization"].iloc[0]
    reduction_technique = scores_df["dimensionality_reduction"].iloc[0]
    return vectorization_technique, reduction_technique


def extract_unique_reduction_settings(scores_df):
    reduction_settings = scores_df["dim_setting"].unique()
    return reduction_settings


def create_dataframe_with_scores_columns():
    df = pd.DataFrame(
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
    return df


def calculate_mean_over_same_setting_rows(scores_df, dim, dim_setting, cl_setting):
    same_settings_df = find_same_setting_rows(scores_df, dim, dim_setting, cl_setting)
    mean_df = same_settings_df.mean()
    return mean_df


def find_same_setting_rows(scores_df, dim, dim_setting, cl_setting):
    same_settings_df = scores_df[
        (scores_df["dim"] == dim)
        & (scores_df["dim_setting"] == dim_setting)
        & (scores_df["clustering_setting"] == cl_setting)
    ]
    return same_settings_df


def create_appendable_dataframe_with_technique_names(
    mean_df, vectorization_technique, reduction_technique, clustering_technique
):
    appendable_df = create_dataframe_with_scores_columns()
    appendable_df = appendable_df.append(mean_df, ignore_index=True)
    appendable_df["vectorization"] = vectorization_technique
    appendable_df["dimensionality_reduction"] = reduction_technique
    appendable_df["clustering"] = clustering_technique
    return appendable_df


def append_to_all_mean_df(mean_df, all_mean_df):
    all_mean_df = pd.concat([all_mean_df, mean_df])
    return all_mean_df


def set_floats_to_ints_after_mean_calculations(df):
    df["dim"] = df["dim"].astype(int)
    df["dim_setting"] = df["dim_setting"].astype(int)
    df["clustering_setting"] = df["clustering_setting"].astype(int)
    return df


def prepare_for_heatmap(scores_df, main_value):
    vectorization_technique, _ = extract_techniques_string_from_dataframe(scores_df)
    hm_df = create_heatmap_dataframe(scores_df, vectorization_technique, main_value)
    # hm_df = attatch_heatmap_dataframe_plot_index(hm_df)
    hm_df = set_heatmap_column_types(hm_df, main_value)
    hm_df, settings_ordered = pivot_and_sort_heatmap(hm_df, main_value)
    return hm_df, settings_ordered


def create_heatmap_dataframe(scores_df, vectorization_technique, main_value):
    hm_df = pd.DataFrame(columns=["setting", "dim", main_value])
    for _, row in scores_df.iterrows():
        add_df = combine_settings_into_string(row, vectorization_technique, main_value)
        hm_df = pd.concat([hm_df, add_df])
    return hm_df


def set_heatmap_column_types(hm_df, main_value):
    hm_df["setting"] = hm_df["setting"].astype("category")
    hm_df["dim"] = hm_df["dim"].astype("int")
    hm_df[main_value] = hm_df[main_value].astype("float")
    return hm_df


def combine_settings_into_string(row, vectorization_technique, main_value):
    add_df = pd.DataFrame(
        [
            [
                int(row["dim"]),
                vectorization_technique
                + "_"
                + row["clustering"]
                + str(int(row["clustering_setting"]))
                + "_"
                + row["dimensionality_reduction"]
                + str(int(row["dim_setting"])),
                row[main_value],
            ]
        ],
        columns=["dim", "setting", main_value],
    )
    return add_df


def plot_heatmap(hm_df, settings_ordered):
    print(hm_df)
    plt.figure()
    sns.heatmap(data=hm_df, yticklabels=settings_ordered, annot=True)
    plt.show()


def pivot_and_sort_heatmap(hm_df, main_value):
    settings_ordered = hm_df["setting"].unique()
    hm_df = hm_df.pivot("setting", "dim", main_value)
    hm_df = hm_df.reindex(settings_ordered)
    return hm_df, settings_ordered


def get_x_y_labels_for_heatmap(hm_df):
    x_labels = hm_df["dim"].unique()
    x_labels.sort()
    y_labels = hm_df["setting"].unique()
    return x_labels, y_labels


def create_csvs(scores_paths):
    metrics = [
        "rand_score",
        "mutual_score",
        # "silhouette",
        # "calinski_harabazs",
        # "davies_bouldin",
    ]

    for metric_index, metric in enumerate(metrics):
        for score_path in scores_paths:
            settings, _, _ = find_settings_from_score_path(score_path)
            scores_df = pd.read_pickle(score_path)
            mean_df = calculate_mean_scores_over_all_settings(scores_df, settings)
            hm_df, settings_ordered = prepare_for_heatmap(mean_df, metric)
            save_path = create_save_path(score_path, metric_index)
            print(save_path)
            hm_df.to_csv(save_path)


def create_save_path(score_path, metric_index):
    save_metrics = ["ari", "ami"]  # , "si", "mri", "nr"
    save_path = score_path.replace("scores/scores_", "csv/").replace(".pkl", "_")
    save_path = save_path + save_metrics[metric_index] + ".csv"
    return save_path


def plot_dataset_scores_for_vectorization(scores_paths, plot_title="BERT SNACK PCA"):
    set_seaborn_style()

    fig, ax = create_plot_axes()
    fig.suptitle(plot_title)

    for i, scores_path in enumerate(scores_paths):
        settings, title, name = find_settings_from_score_path(scores_path)
        scores_df = pd.read_pickle(scores_path)
        k_means_settings, hdbscan_settings = extract_k_means_and_hdbscan_settings(
            settings
        )
        average_df = calculate_mean_scores_over_clustering_settings(scores_df, settings)
        print(average_df)
        average_df = convert_dataframe_column_from_float_to_category(average_df, "dim")
        average_df = convert_dataframe_setting_column_from_ordered_settings(
            average_df, "clustering_setting", hdbscan_settings + k_means_settings
        )
        for j, numeric_score in enumerate(
            ["topic_coherence", "topic_diversity"]
        ):  # "calinski_harabazs", "davies_bouldin", "rand_score", "mutual_score"
            position = [j, i]
            seaborn_lineplot_on_axis(average_df, ax, position, numeric_score)
            remove_axis_clutter(ax, position)
        set_legend_for_the_i_th_axis(ax, i)
        set_ari_and_ami_labels(ax)
        set_title_for_i_th_axis(ax, i, title)
    set_xlabel_and_xticks(ax)
    fig.savefig("scores_plot", bbox_inches="tight")
    plt.show()


def set_seaborn_style():
    sns.set()
    sns.set_theme(style="whitegrid")
    sns.color_palette("coolwarm", as_cmap=True)


def create_plot_axes():
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.5))
    return fig, axes


def seaborn_lineplot_on_axis(df, axes, position, numeric_score, focus="setting"):
    [j, i] = position
    sns.lineplot(
        data=df,
        ax=axes[j, i],
        x="dim",
        y=numeric_score,
        hue=focus,
        style=focus,
        markers=True,
    )


def set_legend_for_the_i_th_axis(axis, i):
    axis[1, i].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        fancybox=True,
        shadow=True,
        ncol=3,
        fontsize=9,
    )


def set_ari_and_ami_labels(ax):
    ax[0, 0].set_ylabel("Coherence")  # ARI
    ax[1, 0].set_ylabel("Diversity")  # AMI


def set_title_for_i_th_axis(ax, i, title):
    ax[0, i].set_title(title)


def set_xlabel_and_xticks(ax):
    plt.setp(ax[:, :], xlabel="Dimensions")
    plt.setp(ax[:, :], xticks=DIMENSIONS, xticklabels=DIMENSIONS)


def remove_axis_clutter(axes, position):
    [j, i] = position
    axes[j, i].semilogx()
    # axes[j, i].get_legend().remove()
    axes[j, i].set_xlabel("")
    axes[j, i].set_ylabel("")


def find_settings_from_score_path(name):
    name = re.sub("scores/scores_", "", name)
    name = re.sub(".pkl", "", name)
    if "ag" in name:
        settings = AG_SETTINGS
        title = "AG"
    elif "snack" in name:
        settings = SNACK_SETTINGS
        title = "SNACK"
    elif "reuters" in name or "_r_":
        settings = R_SETTINGS
        title = "REUTERS"
    else:
        raise Exception("File name wrong. Could not find ag, snack or reuters in path.")
    return settings, title, name


def find_settings_from_title(title):
    if title == "AG":
        return AG_SETTINGS, "AG"
    elif title == "SNACK":
        return SNACK_SETTINGS, "SNACK"
    elif title == "REUTERS":
        return R_SETTINGS, "REUTERS"
    else:
        raise Exception("File name wrong. Could not find ag, snack or reuters in path.")


def plot_clustering_time(
    dfs,
    save_path="time_test",
    show_plot=True,
):
    plot_df = pd.DataFrame()
    for i, df in enumerate(dfs):
        print(i)
        name = df["name"].loc[0]
        save_path = "figures/time/test_" + name
        settings, title = find_settings_from_title(df["title"].loc[0])

        name_setting = re.sub("_\d+b", "", name)
        df["setting"] = name_setting
        df = df[df["clustering_setting"] == 10]
        k_means_settings = settings[1]
        hdbscan_settings = settings[2]
        convert_dataframe_column_from_float_to_category(df, "dim")
        plot_df = plot_df.append(df, ignore_index=True)
        print(plot_df)
    seaborn_lineplot_clustering_time(plot_df, save_path, show_plot)


def seaborn_lineplot_clustering_time(print_df, save_path="time_comp", show_plot=True):
    plt.figure()
    sns.set_theme(style="ticks")
    sns.lineplot(data=print_df, markers=True)
    plt.xscale("log")
    plt.ylabel("Clustering Time (s)")
    plt.xlabel("Dimensions")
    plt.grid()
    plt.xticks(DIMENSIONS, DIMENSIONS)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=5,
        fontsize=9,
    )
    plt.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()


def plot_dim_reduction_time(
    umap_paths,
    pca_paths,
    save_path="figures/time_dim_red.png",
    show_plot=True,
):
    all_df = pd.DataFrame()
    for path_to_df in umap_paths:
        df = pd.read_pickle(path_to_df)
        all_df = all_df.append(df, ignore_index=True)
    umap_df = calculate_mean_scores_over_dimensions(all_df)
    umap_df = convert_dataframe_column_from_float_to_category(umap_df, "dim")
    umap_df["technique"] = "umap"
    umap_df["time (s)"] = umap_df["time_dim_red"]

    all_df = pd.DataFrame()
    for path_to_df in pca_paths:
        df = pd.read_pickle(path_to_df)
        all_df = all_df.append(df, ignore_index=True)
    pca_df = calculate_mean_scores_over_dimensions(all_df)
    pca_df = convert_dataframe_column_from_float_to_category(pca_df, "dim")
    pca_df["technique"] = "pca"
    pca_df["time (s)"] = pca_df["time_dim_red"]

    all_df = pd.DataFrame()
    for path in umap_paths + pca_paths:
        df = pd.read_pickle(path_to_df)
        all_df = all_df.append(df, ignore_index=True)
    cluster_df = calculate_mean_scores_over_clustering_methods(all_df, "")
    cluster_df = convert_dataframe_column_from_float_to_category(cluster_df, "dim")
    cluster_df["technique"] = cluster_df["clustering"]
    cluster_df["time (s)"] = cluster_df["time_clustering"]

    plot_df = umap_df.append(
        pca_df.append(cluster_df, ignore_index=True), ignore_index=True
    )
    print(plot_df)
    seaborn_lineplot_dim_red_time(plot_df, save_path, show_plot)


def seaborn_lineplot_dim_red_time(print_df, save_path="time_comp", show_plot=True):
    plt.figure(figsize=(4, 3))
    sns.set_theme(style="ticks")
    sns.lineplot(
        data=print_df,
        markers=True,
        markersize=8,
        style="technique",
        hue="technique",
        x="dim",
        y="time (s)",
    )
    plt.xscale("log")
    plt.ylabel("Time (s)")
    plt.xlabel("Dimensions")
    plt.grid()
    plt.xticks(DIMENSIONS, DIMENSIONS)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        fancybox=True,
        shadow=True,
        ncol=5,
        fontsize=9,
    )
    plt.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()


def plot_compact_time(paths_to_dfs):
    plot_df = add_time_scores_to_plot_df(paths_to_dfs)
    seaborn_lineplot_time(plot_df)


def add_time_scores_to_plot_df(paths_to_dfs):
    dfs = []
    plot_df = pd.DataFrame()
    for path in paths_to_dfs:
        df = pd.read_pickle(path)
        settings, title, name = find_settings_from_score_path(path)
        average_df = calculate_mean_scores_over_clustering_settings(df, settings)
        average_df["title"] = title
        average_df["name"] = name
        dfs.append(average_df)
        plot_df = plot_df.append(average_df, ignore_index=True)
    plot_df = convert_dataframe_column_from_float_to_category(plot_df, "dim")
    plot_df = convert_dataframe_column_from_float_to_category(
        plot_df, "clustering_setting"
    )
    return plot_df


def seaborn_lineplot_time(plot_df):
    sns.color_palette()
    plt.figure()
    sns.lineplot(
        data=plot_df,
        hue="clustering_setting",
        x="dim",
        y="time_clustering",
        err_style="band",
    )
    plt.xscale("log")
    plt.xticks(DIMENSIONS, DIMENSIONS)
    plt.show()


def plot_n_neighbors_comparison(scores_paths):
    plot_df, labels = add_mean_umap_scores_to_plot_df(scores_paths)
    plot_df = convert_dataframe_column_from_float_to_category(plot_df, "df_index")
    plot_df = convert_dataframe_column_from_float_to_category(plot_df, "dim_setting")
    x_ticks_ints = plot_df["dim_setting"].to_list()
    pivot_df = plot_df.pivot("dim_setting", "df_index", "rand_score")
    seaborn_lineplot_neighbor_comp(pivot_df, labels, x_ticks_ints, ylabel="ARI")


def add_mean_umap_scores_to_plot_df(scores_paths):
    plot_df = pd.DataFrame()
    labels = []
    for i, path in enumerate(scores_paths):
        df = pd.read_pickle(path)
        settings, _, name = find_settings_from_score_path(path)
        name = re.sub("_\d+b", "", name)
        labels.append(name)
        for umap_setting in settings[UMAP_SETTING_POSITION]:
            average_df = df[df["dim_setting"] == umap_setting]
            average_df = average_df.mean()
            average_df["df_index"] = i
            plot_df = plot_df.append(average_df, ignore_index=True)
    return plot_df, labels


def seaborn_lineplot_neighbor_comp(df, labels, x_ticks_ints, ylabel):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(4, 2.5))
    sns.lineplot(data=df, markers=True)
    plt.xscale("log")
    plt.xticks([])
    plt.xticks(x_ticks_ints, map(str, x_ticks_ints), rotation=90)
    plt.xlabel("n_neighbors")
    plt.ylabel(ylabel)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fancybox=True,
        shadow=True,
        ncol=3,
        labels=labels,
    )
    plt.savefig("test", bbox_inches="tight")
    plt.show()


def plot_neighbor_boxplot(paths_to_dfs):
    plot_df = add_neighbor_scores_into_plot_df(paths_to_dfs)
    seaborn_boxplot_neighbor_boxplot(plot_df)


def add_neighbor_scores_into_plot_df(paths_to_dfs):
    plot_df = pd.DataFrame()
    for path in paths_to_dfs:
        df = pd.read_pickle(path)
        settings, title, name = find_settings_from_score_path(path)
        average_df = calculate_mean_scores_over_clustering_settings(df, settings)
        average_df = remove_rows_with_k_means_scores(average_df)
        plot_df = plot_df.append(average_df, ignore_index=True)
    plot_df = convert_dataframe_column_from_float_to_category(
        plot_df, "clustering_setting"
    )
    plot_df = plot_df.dropna()
    return plot_df


def remove_rows_with_k_means_scores(average_df):
    for k in SNACK_KMEANS_SETTINGS + AG_KMEANS_SETTINGS + R_KMEANS_SETTINGS:
        average_df = average_df[average_df["clustering_setting"] != k]
    return average_df


def seaborn_boxplot_neighbor_boxplot(plot_df):
    print(plot_df["clustering_setting"].unique())
    plt.figure()
    sns.boxplot(
        data=plot_df, x="clustering_setting", y="topic_diversity", showfliers=False
    )
    plt.xlabel("min_cluster_size")
    plt.ylabel("Number of clusters")
    plt.show()


def convert_dataframe_column_from_float_to_category(df, column_name):
    df[column_name] = df[column_name].astype("int")
    df[column_name] = df[column_name].astype("category")
    return df


def convert_dataframe_setting_column_from_ordered_settings(df, column, order):
    df[column] = df[column].astype(int)
    df["setting"] = pd.Categorical(
        df[column].astype("str"),
        map(str, order),
    )
    print(df)
    return df


def remove_outliers_in_2D_plot(vectors, threshold):
    mean_vec = np.zeros(len(vectors[0]))
    for v in vectors:
        mean_vec = mean_vec + v
    mean_vec = mean_vec / len(vectors)

    dist_vectors = np.array([np.linalg.norm(v - mean_vec) for v in vectors])
    vectors = vectors[dist_vectors < 11]
    labels = labels[dist_vectors < 11]
    return vectors, labels


def plot_trends(
    scores_paths,
    to_plot="c_v",
    plot_title="SNACK",
    save_path="figures/trends_plot.png",
    show_legend=True,
):
    plot_df = pd.DataFrame()
    for i, scores_path in enumerate(scores_paths):
        settings, title, name = find_settings_from_score_path(scores_path)
        scores_df = pd.read_pickle(scores_path)
        k_means_settings, hdbscan_settings = extract_k_means_and_hdbscan_settings(
            settings
        )
        plot_string = (
            scores_df["vectorization"].loc[0]
            + "_"
            + scores_df["dimensionality_reduction"].loc[0]
        )
        average_df = calculate_mean_scores_over_clustering_methods(
            scores_df, plot_string
        )
        average_df = convert_dataframe_column_from_float_to_category(average_df, "dim")
        plot_df = plot_df.append(average_df, ignore_index=True)
    plot_df = plot_df.rename(columns={"topic_coherence": "c_v"})
    plot_df = plot_df.rename(columns={"rand_score": "ARI"})
    plot_df = plot_df.rename(columns={"mutual_score": "AMI"})

    print(plot_df)
    set_seaborn_style()
    plt.figure(figsize=(480 * px, 400 * px))
    ax = sns.lineplot(
        data=plot_df,
        x="dim",
        y=to_plot,
        hue="configuration",
        style="configuration",
        markers=True,
        markersize=7,
        palette=sns.color_palette("Paired"),
    )
    plt.xscale("log")
    plt.xticks(DIMENSIONS, DIMENSIONS)
    if show_legend:
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=True,
            shadow=True,
            ncol=4,
        )
    else:
        ax.get_legend().remove()
    plt.title(plot_title)
    plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_nr_clusters_vs_metric(
    scores_path,
    metric="ARI",
    plot_title="SNACK",
    save_path="figures/_clusters.png",
    show_legend=True,
):
    plot_df = pd.DataFrame()
    for scores in scores_path:
        add_df = pd.read_pickle(scores)
        add_df["configuration"] = (
            add_df["vectorization"]
            + "_"
            + add_df["dimensionality_reduction"]
            + "_"
            + add_df["clustering"]
        )
        plot_df = plot_df.append(add_df, ignore_index=True)
    plot_df = plot_df.rename(columns={"topic_coherence": "c_v"})
    plot_df = plot_df.rename(columns={"rand_score": "ARI"})
    plot_df = plot_df.rename(columns={"mutual_score": "AMI"})
    set_seaborn_style()
    plt.figure(figsize=(480 * px, 400 * px))
    ax = sns.scatterplot(
        data=plot_df,
        x="nr_clusters",
        y=metric,
        hue="configuration",
        palette=sns.color_palette("Paired"),
        s=4,
    )
    if show_legend:
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=True,
            shadow=True,
            ncol=4,
        )
    else:
        ax.get_legend().remove()
    plt.xscale("log")
    plt.title(plot_title)
    plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_time_vs_dim(scores_paths, time="time_dim_red"):

    for score in scores_paths:
        df = pd.read_pickle(score)


def print_top_settings_for_combination(scores_path, value="ARI"):
    df = pd.read_pickle(scores_path)
    df["configuration"] = (
        df["vectorization"]
        + "_"
        + df["dimensionality_reduction"]
        + "_"
        + df["clustering"]
    )
    df["setting_string"] = ""
    for i, row in df.iterrows():
        row["setting_string"] = (
            str(row["dim"])
            + "_"
            + str(row["dim_setting"])
            + "_"
            + str(row["clustering_setting"])
        )

    mean_df = pd.DataFrame()
    for setting_string in df["setting_string"].unique():
        add_df = df[df.setting_string == setting_string]
        clustering_string = add_df["clustering"].iloc[0]
        add_df = add_df.mean()
        add_df["clustering"] = clustering_string
        mean_df = mean_df.append(add_df, ignore_index=True)
    df = mean_df
    df = df.rename(columns={"topic_coherence": "c_v"})
    df = df.rename(columns={"rand_score": "ARI"})
    df = df.rename(columns={"mutual_score": "AMI"})
    configurations = df["clustering"].unique()
    for config in configurations:
        print_df = df[df["clustering"] == config]
        print_df = print_df.sort_values(by=value)
        print(len(print_df))
        print(print_df.tail(5))


"""
Dim vectors:
bert_reuters.pkl
bert_ag.pkl
bert_snack.pkl

doc2vec_reuters.pkl
doc2vec_ag.pkl
doc2vec_snack.pkl

"""


def main():
    # PLOT 2D
    # corpus_df = pd.read_pickle("dim_vectors/doc2vec_reuters.pkl")
    # doc_vec = corpus_df["doc_vec"].to_list()

    # # vectors = reduce_umap(doc_vec, 2, 20)
    # vectors = reduce_pca(doc_vec, 2)

    # plot_2D(corpus_df, vectors, labels=corpus_df["label"].to_list(), dataset_identifier="REUTERS", plot_title="REUTERS Doc2Vec PCA", crop=[2, -2, 2, -2])

    # SNACK BERT UMAP [20, 2, 15, -15]
    # SNACK Doc2Vec UMAP [5, -5, 15, -10]
    # AG BERT UMAP [12, -20, 30, -20]
    # REUTERS BERT UMAP None
    # REUTERS Doc2Vec PCA [2, -2, 2, -2]
    # REUTERS Doc2Vec AG [10, -10, 10, -10]

    # # # PLOT CLUSTERED LABELS
    # scores_paths = [
    #     "scores/scores_bert_r_230119.pkl",
    #     "scores/scores_d2v_r_230111.pkl",
    #     "scores/scores_bert_r_230119_pca.pkl",
    #     "scores/scores_d2v_r_230111_pca.pkl",
    # ]
    # # # plot_dataset_scores_for_vectorization(scores_paths, "BERT PCA")
    # # plot_trends(scores_paths, "c_v", "AG NEWS", show_legend=False)

    # # # PLOT NR CLUSTERS VS METRIC
    # plot_nr_clusters_vs_metric(scores_paths, "AMI", "REUTERS", show_legend=True)

    # # PLOT TIME VS DIMENSION
    # scores_paths = [
    #     "scores/scores_bert_r_230119.pkl",
    #     "scores/scores_d2v_r_230111.pkl",
    #     "scores/scores_bert_r_230119_pca.pkl",
    #     "scores/scores_d2v_r_230111_pca.pkl",
    # ]
    # plot_time_vs_dim(scores_paths, "time_dim_red")

    # # PRINT TOP CONFIGURATIONS
    # scores_path = "scores/scores_d2v_ag_230111.pkl"
    # print_top_settings_for_combination(scores_path, "ARI")


    # # PLOT TIME DIMENSIONALITY REDUCTION
    # umap_paths = [
    #     "scores/scores_bert_r_230119.pkl",
    #     "scores/scores_d2v_r_230111.pkl",
    # ]
    # pca_paths = [
    #     "scores/scores_bert_r_230119_pca.pkl",
    #     "scores/scores_d2v_r_230111_pca.pkl",
    # ]
    # plot_dim_reduction_time(umap_paths, pca_paths)
    pass 


if __name__ == "__main__":
    main()