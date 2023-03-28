from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW
from transformers import TrainingArguments
from transformers import Trainer

import torch
import pandas as pd


class ArticlesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def train_torch_model(corpus_path, save_path):
    corpus_df = pd.read_pickle(corpus_path)
    text = corpus_df["text"].to_list()

    body = ""
    for article in text:
        body = body + " " + article
    splitted_text = body.split(".")
    text = splitted_text
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
    )
    inputs["labels"] = inputs.input_ids.detach().clone()
    rand = torch.rand(inputs.input_ids.shape)
    protected_tokens = (
        (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
    )
    mask_arr = (rand < 0.15) * protected_tokens

    # create selection from mask_arr
    selection = []
    for i in range(inputs.input_ids.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

    # apply selection index to inputs.input_ids, adding MASK tokens
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    # Initialize our data using the MeditationsDataset class.
    dataset = ArticlesDataset(inputs)
    # dataset = torch.utils.data.Dataset(inputs)

    # And initialize the dataloader, which we'll be using to load our data into the model during training.
    # loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    args = TrainingArguments(
        output_dir="bert_output",
        per_device_train_batch_size=6,
        num_train_epochs=2,
        logging_steps=2000,
        save_steps=4000,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    torch.save(model.state_dict(), save_path)


def main():
    train_torch_model()


if __name__ == "__main__":
    main()
