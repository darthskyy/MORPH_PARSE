# %%
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
import os, csv
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Parsing inputs for training the model")
parser.add_argument("--data", type=str, default="data", help="The directory where the data is stored.")
parser.add_argument("--checkpoint", type=str, default="xlm-roberta-base", help="The pretrained checkpoint to use for the model. Must be a model that supports token classification.")
parser.add_argument("--output", type=str, default="xlmr", help="The output directory for the model in the models directory.")
parser.add_argument("--epochs", type=int, default=1, help="The number of epochs to train the model for.")
parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use for training and evaluation.")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="The learning rate to use for training.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay to use for training.")
parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="The evaluation strategy to use for training.")
parser.add_argument("--lang", type=str, default="NR", help="The language to train the model on.", choices=["NR","SS","XH","ZU"])
parser.add_argument("--validation_split", type=float, default=0.1, help="The fraction of the training data to use for validation.")
parser.add_argument("--save_steps", type=int, default=500, help="The number of steps to save the model after.")
parser.add_argument("--save_total_limit", type=int, default=2, help="The total number of models to save.")
args = parser.parse_args()

data_dir = args.data

# load the dataset for the specified language
lang = args.lang
column_names = ["word", "parsed", "morpheme", "tag"]
lang_set = {
    "TRAIN": pd.read_csv(f"{data_dir}/TRAIN/{lang}_TRAIN.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, names=column_names)
    ,
    "TEST": pd.read_csv(f"{data_dir}/TEST/{lang}_TEST.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, names=column_names,)
    ,
}

# split the training data into training and validation sets

lang_set["VAL"] = lang_set["TRAIN"].sample(frac=args.validation_split, random_state=42)
lang_set["TRAIN"] = lang_set["TRAIN"].drop(lang_set["VAL"].index)



# %%
print("loaded the datasets")

# %%
mappings = {}
mappings_r = {}
count = 0
def extract_tag(seq: str) -> str:
    global mappings, count
    seq = seq.split("_")
    for i, tag in enumerate(seq):
        if tag not in mappings.keys():
            mappings[tag] = count
            mappings_r[count] = tag
            count+=1
        seq[i] = mappings[tag]
    return seq

# %%
for item in ["TEST", "TRAIN", "VAL"]:
    df = lang_set[item]
    df['morpheme'] = df['morpheme'].apply(lambda x: x.split("_"))
    df['tag'] = df['tag'].apply(lambda x: extract_tag(x))

# %%
print("mapped the input")

# %%
dataset = {
    "train": Dataset.from_pandas(lang_set["TRAIN"]),
    "test": Dataset.from_pandas(lang_set["TEST"]),
    "validation": Dataset.from_pandas(lang_set["VAL"])
}

lang_set = DatasetDict(dataset)

# %%
print("datasets created")

# %%
from transformers import XLMRobertaTokenizerFast
checkpoint = args.checkpoint
tokenizer = XLMRobertaTokenizerFast.from_pretrained(checkpoint)


# %%
def tokenize_and_align(example, label_all_tokens=True):
    """
    Tokenizes the input example and aligns the labels with the tokenized input.

    Args:
        example (dict): The input example containing the "morpheme" and "tag" fields.
        label_all_tokens (bool, optional): Whether to include labels for all tokens. Defaults to True.

    Returns:
        dict: The tokenized input with aligned labels.

    """
    tokenized_input = tokenizer(example["morpheme"], truncation=True, is_split_into_words=True)
    labels = []

    for i, label in enumerate(example["tag"]):
        word_ids = tokenized_input.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_input["labels"] = labels
    return tokenized_input

# %%
tokenized_dataset = lang_set.map(tokenize_and_align, batched=True)

# %%
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(mappings))

# %%
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir=args.output,
    evaluation_strategy=args.evaluation_strategy,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    logging_dir=args.output+"/logs",
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit
)

# %%
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer)

from datasets import load_metric

metric = load_metric("seqeval")

# %%
import numpy as np
def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds
    pred_logits = np.argmax(pred_logits, axis=2)

    predictions = [
        ["_-"+mappings_r[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        ["_-"+mappings_r[l] for (eval_preds, l) in zip(prediction, label) if l != -100] for prediction, label in zip(pred_logits, labels)
    ]

    results = metric.compute(predictions=predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"] ,
        "accuracy": results["overall_accuracy"],
    }

# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.train()

# %%
model.save_pretrained("parse_model")
tokenizer.save_pretrained("tokenizer")

# %%
import json

config = json.load(open("parse_model/config.json"))
config["id2label"] = mappings_r
config["label2id"] = mappings

json.dump(config, open("parse_model/config.json", "w"))


