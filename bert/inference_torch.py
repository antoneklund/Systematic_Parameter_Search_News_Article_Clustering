import os
import sys

sys.path.append(os.getcwd())  # noqa

from transformers import BertForMaskedLM, AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel

import transformers
import torch
import torch.nn.functional as F

import pandas as pd

PATH = os.getcwd() + "/models/bert.torch"


def small_bert_inference(corpus_df, save_path="dim_vectors/small_bert_trained.pkl"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained(
        "google/bert_uncased_L-4_H-512_A-8", output_hidden_states=True
    )
    model.eval()

    corpus_df["doc_vec"] = None
    for i, row in corpus_df.iterrows():
        inputs = tokenizer(
            row["text"],
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        outputs = model(**inputs)
        output_avg = outputs.hidden_states[1:-1][0][0].detach().numpy()
        # print(output_avg.shape)
        corpus_df["doc_vec"].iloc[i] = output_avg.mean(axis=0)
        print(corpus_df.iloc[i])

    print(corpus_df)
    corpus_df.to_pickle(save_path)


def trained_bert_inference(
    corpus_df, model_path=PATH, save_path="dim_vectors/bert_trained.pkl"
):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained(
        "bert-base-uncased", output_hidden_states=True
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    corpus_df["doc_vec"] = None
    for i, row in corpus_df.iterrows():
        inputs = tokenizer(
            row["text"],
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        outputs = model(**inputs)
        output_avg = outputs.hidden_states[-1][0].detach().numpy()
        output_avg = output_avg[1:]
        # print(output_avg)
        corpus_df["doc_vec"].iloc[i] = output_avg.mean(axis=0)
        print(corpus_df.iloc[i])

    print(corpus_df)
    corpus_df.to_pickle(save_path)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def inference_sentence_bert(
    corpus_df, model_path=PATH, save_path="dim_vectors/sbert_vectors.pkl"
):
    # corpus_df = corpus_df.sample(1000)
    sentences = corpus_df["text"].to_list()
    print(len(sentences))

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize sentences

    embeddings = []
    for i, sentence in enumerate(sentences):
        encoded_input = tokenizer(
            sentence, padding=True, truncation=True, return_tensors="pt"
        )
        # print("tokenization done")
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # print("Sentence embeddings:")
        print(i)
        embeddings.append(sentence_embeddings.numpy()[0])

    corpus_df["doc_vec"] = embeddings
    corpus_df.to_pickle(save_path)


def inference_bert_uncased(corpus_df, save_path="dim_vectors/bert_vectors.pkl", model_path=""):
    texts = corpus_df["text"].to_list()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if model_path != "":
        model.load_state_dict(torch.load(model_path))
        model.eval()

    model.to(device)

    embeddings = []
    for i, text in enumerate(texts):
        encoded_input = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        encoded_input.to(device)
        attention_mask = encoded_input.attention_mask  # (1, 512)
        attention_mask[0][0] = 0  # Remove the CLS token
        outputs = model(**encoded_input)
        pooled_output = torch.sum(
            attention_mask.unsqueeze(-1) * outputs.last_hidden_state, dim=1
        ) / torch.unsqueeze(torch.sum(attention_mask, dim=-1), dim=-1)
        pooled_output.to("cpu")
        embeddings.append(pooled_output.detach().numpy()[0])
        # print(attention_mask.shape)
        # print(pooled_output.shape)
        print("Done {}".format(i) + " out of {}".format(len(texts)))
    corpus_df["doc_vec"] = embeddings
    corpus_df.to_pickle(save_path)


def main():
    corpus_df = pd.read_pickle("datasets/SNACK_RAW.pkl")
    # corpus_df = corpus_df.sample(20)
    corpus_df["text"] = corpus_df["body"]

    # Load model
    inference_bert_uncased(corpus_df, save_path="dim_vectors/test_new_vectors.pkl")


if __name__ == "__main__":
    main()
