# %%
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
import os, csv
import pandas as pd
import numpy as np
import argparse
import warnings
import time

# * for timing purposes
absolute_start = time.time()
def format_time(t):
    """
    Formats the time in seconds to a human readable format.

    Args:
        t (float): The time in seconds.

    Returns:
        str: The formatted time in hours, minutes, and seconds.
    """
    h = int(t//3600)
    m = int((t%3600)//60)
    s = int(t%60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def log_message(message):
    """
    Logs a message to the console with the time elapsed since the start of the script.

    Args:
        message (str): The message to log.
    """
    print(f"{format_time(time.time()-absolute_start)}\t-\t{message}")
# * load the arguments from the command line
parser = argparse.ArgumentParser(description="Parsing inputs for training the model")
# model data, loading and saving arguments
parser.add_argument("--data", type=str, default="../data", help="The directory where the data is stored.")
parser.add_argument("--lang", type=str, default="NR", help="The language to train the model on.", choices=["NR","SS","XH","ZU"])
parser.add_argument("--checkpoint", type=str, default="xlm-roberta-base", help="The pretrained checkpoint to use for the model. Must be a model that supports token classification.")
parser.add_argument("--resume_from_checkpoint", action="store_true", help="Whether to resume training from a checkpoint.")
parser.add_argument("--output", type=str, default="xlmr", help="The output directory for the model in the models directory.")

# training arguments
parser.add_argument("--seed", type=int, default=42, help="The seed to use for reproducibility.")
parser.add_argument("--epochs", type=int, default=1, help="The number of epochs to train the model for.")
parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use for training and evaluation.")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="The learning rate to use for training.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay to use for training.")
parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="The evaluation strategy to use for training.")
parser.add_argument("--validation_split", type=float, default=0.1, help="The fraction of the training data to use for validation.")
parser.add_argument("--save_steps", type=int, default=500, help="The number of steps to save the model after.")
parser.add_argument("--save_total_limit", type=int, default=2, help="The total number of models to save.")
parser.add_argument("--metric", type=str, default="all", help="The metric to use for evaluation.", choices=["all", "f1", "precision", "recall"])

# training flags
parser.add_argument("--load_best_model_at_end", type=bool, default=True, help="Whether to load the best model at the end of training.")
parser.add_argument("--metric_for_best_model", type=str, default="loss", help="The metric to use for the best model.")
parser.add_argument("--greater_is_better", type=bool, default=False, help="Whether a greater value of the metric is better.")
parser.add_argument("--warning", type=bool, default=False, help="Whether to show warnings or not.")

# debugging and logging
parser.add_argument("--f", type=str, default="morpheme", help="The field to use for the morphemes.")
parser.add_argument("--debug", type=bool, default=True, help="Whether to run the script in debug mode.")
parser.add_argument("--log", type=str, default="train.log", help="The log file to write to.")
parser.add_argument("--verbose", type=bool, default=True, help="Whether to show verbose output or not.")

args = parser.parse_args()

# * show warnings if the warning flag is set
if not args.warning:
    warnings.filterwarnings("ignore")

# * load the dataset

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

lang_set["VAL"] = lang_set["TRAIN"].sample(frac=args.validation_split, random_state=args.seed)
lang_set["TRAIN"] = lang_set["TRAIN"].drop(lang_set["VAL"].index)


log_message("loaded the datasets")

# %%
# * map the rags to corresponding integers
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

for item in ["TEST", "TRAIN", "VAL"]:
    df = lang_set[item]
    df['morpheme'] = df['morpheme'].apply(lambda x: x.split("_"))
    df['tag'] = df['tag'].apply(lambda x: extract_tag(x))

log_message("mapped the input")

# %%
# * create the dataset
dataset = {
    "train": Dataset.from_pandas(lang_set["TRAIN"]),
    "test": Dataset.from_pandas(lang_set["TEST"]),
    "validation": Dataset.from_pandas(lang_set["VAL"])
}

lang_set = DatasetDict(dataset)
log_message("datasets created")

# %%
# * loading the tokenizer and model
load_start = time.time()
from transformers import XLMRobertaTokenizerFast, AutoModelForTokenClassification
checkpoint = args.checkpoint
tokenizer = XLMRobertaTokenizerFast.from_pretrained(checkpoint, cache_dir=".cache")
model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(mappings), cache_dir=".cache")

log_message(f"loaded the model and tokenizer in {format_time(time.time()-load_start)}")
log_message("loaded the model and tokenizer")

# %%
# * tokenizing the input
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

tokenized_dataset = lang_set.map(tokenize_and_align, batched=True)

log_message("tokenized the dataset")

# %%
# * loading the arguments and training the model
from transformers import TrainingArguments

train_args = TrainingArguments(
    output_dir=args.output,
    evaluation_strategy=args.evaluation_strategy,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    logging_dir=args.output+"/logs",
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    disable_tqdm=True,
    load_best_model_at_end=args.load_best_model_at_end,
    metric_for_best_model=args.metric_for_best_model,
    greater_is_better=args.greater_is_better,
    resume_from_checkpoint=args.resume_from_checkpoint,
    seed=args.seed,
)

# print all the entered args
print(f"output directory: {train_args.output_dir}")
print(f"logging directory: {train_args.logging_dir}")
print(f"evaluation strategy: {train_args.evaluation_strategy}")
print(f"learning rate: {train_args.learning_rate}")
print(f"epochs: {train_args.num_train_epochs}")
print(f"weight decay: {train_args.weight_decay}")
print(f"save steps: {train_args.save_steps}")
print(f"save total limit: {train_args.save_total_limit}")
print(f"per device train batch size: {train_args.per_device_train_batch_size}")
print(f"per device eval batch size: {train_args.per_device_eval_batch_size}")


# %%
# * defining the compute metrics function
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer)
from datasets import load_metric
import numpy as np

metric = load_metric("seqeval")
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
# * adding the trainer
from transformers import Trainer
model = AutoModelForTokenClassification.from_pretrained("models/xlm-roberta-large_NR/checkpoint-17500", num_labels=len(mappings), cache_dir=".cache")
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

log_message("added the trainer")

# %%
# * saving the model, tokenizer and mappings
import json
args.output = "models/xlm-roberta-large_NR/checkpoint-17500"
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)

config = json.load(open(f"{args.output}/config.json"))
config["id2label"] = mappings_r
config["label2id"] = mappings

model.config.id2label = mappings_r
model.config.label2id = mappings

json.dump(config, open(f"{args.output}/config.json", "w"))

# %%
# * training the model
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# %%
# * evaluating the model on the test set
log_message("evaluating the model on the test set: run 1")
x = trainer.evaluate(tokenized_dataset["test"])

print(x)
log_message("evaluation complete")

# * testing the model
from transformers import pipeline
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

log_message("Creating the pipeline")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
log_message("Pipeline created")

# %%
# * formatting the NER results
def format_ner_results(ner_results, model="xlmr"):
    """
    Format the NER results to be used for evaluation

    Args:
    ner_results (list of dictionaries): The NER results containing word and entity information.

    Returns:
    tuple: A tuple containing two lists - morphs and tags. Morphs is a list of morphemes extracted from the NER results, and tags is a list of corresponding entity tags.

    Example:
    >>> ner_results = [
            {"word": "U", "Entity": "NPrePre15"},
            {"word": "ku", "Entity": "BPre15"},
            {"word": "eng", "Entity": "VRoot"},
            {"word": "##ez", "Entity": "VRoot"},
            {"word": "a", "Entity": "VerbTerm"}
        ]
    >>> format_ner_results(ner_results)
    (["u", "ku", "engez", "a"], ["NPrePre15", "BPre15", "VRoot", "VerbTerm"])
    """
    morphs = []
    tags = []

    if model=="xlmr":
        for i in range(len(ner_results)):
            morph = ner_results[i]["word"]
            tag = ner_results[i]["entity"]

            if morph.startswith("▁"):
                morphs.append(morph[1:])
                if "Dem" in tag:
                    continue
                tags.append(tag)
            else:
                morphs[-1] += morph
    elif model=="bert":
        for i in range(len(ner_results)):
            morph = ner_results[i]["word"]
            tag = ner_results[i]["entity"]

            if morph.startswith("##"):
                morphs[-1] += morph[2:]
            else:
                morphs.append(morph)
                if "Dem" in tag:
                    continue
                tags.append(tag)
    
    return morphs, tags

# %%
# * predicting the tags for the test set
test_set = lang_set["test"]
references = []
predictions = []

log_message("Predicting the tags for the test set")
for i in range(len(test_set)):
    sentence = " ".join(test_set[i]["morpheme"])
    ner_results = nlp(sentence)
    morphs, tags = format_ner_results(ner_results)
    expected_tags = ["_-" + mappings_r[x] for x in test_set[i]["tag"]]
    tags = ["_-" + x for x in tags]
    if len(expected_tags) != len(tags):
        continue
    predictions.append(tags)
    references.append(expected_tags)

log_message("Predictions complete")
# %%
# * evaluating the model on the classification report of the test set

log_message("Evaluating the model on the test set: run 2")
if args.metric == "all":
    print(classification_report(references, predictions))
elif args.metric == "f1":
    print(f1_score(references, predictions))
elif args.metric == "precision":
    print(precision_score(references, predictions))
elif args.metric == "recall":
    print(recall_score(references, predictions))

log_message("Evaluation complete")
log_message("Script complete")
