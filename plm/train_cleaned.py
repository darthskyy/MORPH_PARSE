import argparse
import csv
import json
import logging
import os
import sys
import warnings

from datasets import Dataset, DatasetDict
from datasets import load_metric
import numpy as np
import pandas as pd
import torch
from transformers import (XLMRobertaTokenizerFast, AutoModelForTokenClassification, DataCollatorForTokenClassification)
from transformers import (TrainingArguments, Trainer)
from transformers import pipeline
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# * load the arguments from the command line
parser = argparse.ArgumentParser(description="Parsing inputs for training the model")
# model data, loading and saving arguments
parser.add_argument(
    "--server",
    type=str, default="local",
    help="The server to run the script on. If this is set to anything other than local,\
        data directory will be inferred.",
    choices=["local", "uct", "nicis"]
)

parser.add_argument(
    "--data",
    type=str, default="../data",
    help="The directory where the data is stored."
)
parser.add_argument(
    "--lang",
    type=str, default="NR",
    help="The language to train the model on.", choices=["NR","SS","XH","ZU"]
)
parser.add_argument(
    "--checkpoint",
    type=str, default="xlm-roberta-base",
    help="The pretrained checkpoint to use for the model. Must be a model that supports \
        token classification."
)
parser.add_argument(
    "--resume_from_checkpoint",
    action="store_true",
    help="Whether to resume training from a checkpoint."
)
parser.add_argument(
    "--output",
    type=str, default="xlmr",
    help="The output directory for the model in the models directory.")

# training arguments
parser.add_argument(
    "--seed",
    type=int, default=42,
    help="The seed to use for reproducibility."
)
parser.add_argument(
    "--epochs",
    type=int, default=1,
    help="The number of epochs to train the model for."
)
parser.add_argument(
    "--batch_size",
    type=int, default=16,
    help="The batch size to use for training and evaluation."
)
parser.add_argument(
    "--learning_rate",
    type=float, default=2e-5,
    help="The learning rate to use for training."
)
parser.add_argument(
    "--weight_decay",
    type=float, default=0.01,
    help="The weight decay to use for training."
)
parser.add_argument(
    "--evaluation_strategy",
    type=str, default="steps",
    help="The evaluation strategy to use for training.",
    choices=["epoch", "steps"]
)
parser.add_argument(
    "--validation_split",
    type=float, default=0.1,
    help="The fraction of the training data to use for validation."
)
parser.add_argument(
    "--save_steps",
    type=int, default=500,
    help="The number of steps to save the model after."
)
parser.add_argument(
    "--save_total_limit",
    type=int, default=2,
    help="The total number of models to save."
)
parser.add_argument(
    "--metric",
    type=str, default="all",
    help="The metric to use for evaluation.",
    choices=["all", "f1", "precision", "recall"]
)

# training flags
parser.add_argument(
    "--load_best_model_at_end",
    action="store_true",
    help="Whether to load the best model at the end of training."
)
parser.add_argument(
    "--metric_for_best_model",
    type=str, default="loss",
    help="The metric to use for the best model."
)
parser.add_argument(
    "--greater_is_better",
    type=bool, default=False,
    help="Whether a greater value of the metric is better."
)
parser.add_argument(
    "--warning",
    type=bool, default=False,
    help="Whether to show warnings or not.")

# debugging and logging
parser.add_argument(
    "--f",
    type=str, default="morpheme",
    help="The field to use for the morphemes."
)
parser.add_argument(
    "--debug",
    type=bool, default=True,
    help="Whether to run the script in debug mode."
)
parser.add_argument(
    "--log",
    type=str, default="train.log",
    help="The log file to write to."
)
parser.add_argument(
    "--verbose",
    type=bool, default=True,
    help="Whether to show verbose output or not."
)

args = parser.parse_args()

# * checking correctness of the arguments in relation to each other
if args.evaluation_strategy == "steps":
    assert args.save_steps, "The save steps must be specified when using steps evaluation strategy."

if args.load_best_model_at_end:
    assert args.evaluation_strategy == "steps", "The evaluation strategy must be steps when loading the best model at the end."

# * show warnings if the warning flag is set
if not args.warning:
    warnings.filterwarnings("ignore")

# * setting up the logging
logger = logging.getLogger(f"train_{args.lang}_{args.checkpoint}")
logger.setLevel("DEBUG")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)-8s - %(message)s")  # Modified line

# setting up the file handler
fh = logging.FileHandler(args.log)
fh.setLevel("DEBUG")

# setting up the console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel("DEBUG")

# setting up the formatter
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# adding the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

logger.debug("Logging setup complete")
# format the arguments better
def format_args(args):
    out = ""
    for arg in vars(args):
        # put quotes around strings
        if isinstance(getattr(args, arg), str):
            out += f"\t{arg:25}: '{getattr(args, arg)}'\n"
        else:
            out += f"\t{arg:25}: {getattr(args, arg)}\n"
    return out

logger.info(f"\nSetup Arguments Parsed\n{format_args(args)}")

# * load the dataset

if args.server != "local":
    base_dir = os.path.expanduser("~")
    if args.server == "uct":
        suffix = "MORPH_PARSE"
    elif args.server == "nicis":
        suffix = "lustre/MORPH_PARSE"
    args.data = os.path.join(base_dir, suffix)
    args.output = os.path.join(base_dir, suffix, args.output)
    logger.info(f"Server: {args.server}")
    logger.info(f"Inferred data directory: {args.data}")
    logger.info(f"Inferred output directory: {args.output}")
else:
    args.data = os.path.abspath(args.data)
    args.output = os.path.abspath(args.output)
    logger.info(f"Data directory: {args.data}")
    logger.info(f"Output directory: {args.output}")

# * checking for gpu availability
USING_GPU = torch.cuda.is_available()
logger.info(f"Using GPU: {USING_GPU}")

# load the dataset for the specified language
column_names = ["word", "parsed", "morpheme", "tag"]

# TODO add the download data flag
try:
    lang_set = {
        "TRAIN": pd.read_csv(f"{args.data}/TRAIN/{args.lang}_TRAIN.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, names=column_names)
        ,
        "TEST": pd.read_csv(f"{args.data}/TEST/{args.lang}_TEST.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, names=column_names,)
        ,
    }
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    logger.info(f"Files can be found for download at: https://repo.sadilar.org/handle/20.500.12185/546.")
    logger.info("Please download the files and place them in the data directory. (or use the --download_data flag to download the data)")
    sys.exit(1)
# split the training data into training and validation sets

lang_set["VAL"] = lang_set["TRAIN"].sample(frac=args.validation_split, random_state=args.seed)
lang_set["TRAIN"] = lang_set["TRAIN"].drop(lang_set["VAL"].index)


logger.debug("loaded the datasets")
logger.info(f"Training set: {len(lang_set['TRAIN'])}")
logger.info(f"Validation set: {len(lang_set['VAL'])}")
logger.info(f"Test set: {len(lang_set['TEST'])}")


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

logger.debug("mapped the input")
logger.info(f"No. of tags: {len(mappings)}")


# * create the dataset
dataset = {
    "train": Dataset.from_pandas(lang_set["TRAIN"]),
    "test": Dataset.from_pandas(lang_set["TEST"]),
    "validation": Dataset.from_pandas(lang_set["VAL"])
}

lang_set = DatasetDict(dataset)
logger.debug("datasets created")


# * loading the tokenizer and model
checkpoint = args.checkpoint
tokenizer = XLMRobertaTokenizerFast.from_pretrained(checkpoint, cache_dir=".cache")
model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(mappings), cache_dir=".cache")

logger.debug("loaded the model and tokenizer")
logger.info(f"Model: {checkpoint}")

if USING_GPU:
    model.to("cuda")
    logger.info("Model on GPU")

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

logger.debug("tokenized the dataset")


# * loading the arguments and training the model

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
    disable_tqdm=not args.debug,
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



# * defining the compute metrics function
data_collator = DataCollatorForTokenClassification(tokenizer)

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


# * adding the trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

logger.debug("added the trainer")


# * saving the model, tokenizer and mappings
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)

config = json.load(open(f"{args.output}/config.json"))
config["id2label"] = mappings_r
config["label2id"] = mappings

model.config.id2label = mappings_r
model.config.label2id = mappings

json.dump(config, open(f"{args.output}/config.json", "w"))

logger.debug("saved the model, tokenizer and mappings")

# * training the model
# check if the model is to be resumed from a checkpoint
if args.resume_from_checkpoint:
    logger.debug("checking for checkpoint")
    dirs = os.listdir(f"{args.output}")
    if any([x for x in dirs if "checkpoint" in x]):
        resume_points = [x for x in dirs if "checkpoint" in x]
        steps = [int(x.split("-")[1]) for x in resume_points]
        # get the checkpoint with the highest number of steps
        resume_point = resume_points[np.argmax(steps)]
        args.resume_from_checkpoint = f"{args.output}/{resume_point}"
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        logger.info(f"Resuming from step: {max(steps)}")
    else:
        args.resume_from_checkpoint = None
        logger.warning("No checkpoint found. Training from scratch.") 

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


# * evaluating the model on the test set
logger.debug("evaluating the model on the test set: run 1")
x = trainer.evaluate(tokenized_dataset["test"])

logger.debug("evaluation complete")
logger.info(f"Results: {x}")

# * testing the model

logger.debug("Creating the pipeline")
if USING_GPU:
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0, batch_size=16)
else:
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
logger.debug("Pipeline created")


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


# * predicting the tags for the test set
test_set = lang_set["test"]
references = []
predictions = []

logger.debug("Predicting the tags for the test set")
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

logger.debug("Predictions complete")

# * evaluating the model on the classification report of the test set

logger.debug("Evaluating the model on the test set: run 2")
metric = None
if args.metric == "all":
    metric = classification_report
elif args.metric == "f1":
    metric = f1_score
elif args.metric == "precision":
    metric = precision_score
elif args.metric == "recall":
    metric = recall_score

results = metric(references, predictions)
logger.info(f"Results: {results}")

logger.debug("Evaluation complete")
logger.debug("Script complete")
