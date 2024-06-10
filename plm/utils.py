import argparse
import os
import csv


import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
import evaluate
from transformers import XLMRobertaTokenizerFast, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments


class MorphParseDataset():
    def __init__(self, language: str, path: str="data", validation_split: float=0.1, seed: int=42, tokenizer=None) -> None:
        self.language = language
        self.path = path
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large", cache_dir=".cache") \
            if tokenizer is None else tokenizer
        self.train = None
        self.valid = None
        self.test = None

        self.validation_split = validation_split
        self.seed = seed

        self. column_names = ["word", "parsed", "morpheme", "tag"]
        self.id2label = {}
        self.label2id = {}


        self.loaded = False
        self.tokenized = False
    
    def load(self):
        """
        Loads the dataset from the given path.

        Args:
            path (str): The path to the dataset.
            The dataset should be in the following format: 
            path/
            ├── TEST/
            │   ├── {lang)_TEST.tsv
            ├── TRAIN/
            │   ├── {lang}_TRAIN.tsv
            {
            validation to be added later but right now is assumed to be 10% of the training data
            ├── VALID/
            │   ├── {lang}_VALID.tsv
            }

        Returns:
            None
        """
        if self.loaded:
            return

        # first checking if the directory exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Directory {self.path} not found. Please check the path and try again.")

        train_path = os.path.join(self.path, "TRAIN", f"{self.language}_TRAIN.tsv")
        test_path = os.path.join(self.path, "TEST", f"{self.language}_TEST.tsv")
        # valid_path = os.path.join(path, "VALID", f"{self.language}_VALID.tsv")

        train_data = pd.read_csv(train_path, delimiter="\t", names=self.column_names, quoting=csv.QUOTE_NONE)
        test_data = pd.read_csv(test_path, delimiter="\t", names=self.column_names, quoting=csv.QUOTE_NONE)

        # getting the validation data
        validation_data = train_data.sample(frac=self.validation_split, random_state=self.seed)
        train_data.drop(validation_data.index, inplace=True)
        
        # extracting the labels and converting them to ids
        self.extract_labels(pd.concat([train_data, validation_data, test_data]))
        for df in [train_data, validation_data, test_data]:
            df["tag"] = df["tag"].apply(lambda x: [self.label2id[label] for label in x.split("_")])
            df["morpheme"] = df["morpheme"].apply(lambda x: x.split("_"))

        # converting the data to HuggingFace Dataset format
        self.train = Dataset.from_pandas(train_data)
        self.valid = Dataset.from_pandas(validation_data)
        self.test = Dataset.from_pandas(test_data)
        self.loaded = True
    
    def extract_labels(self, data: pd.DataFrame):
        """
        Extracts the labels from the dataset and assigns an id to each label.

        Args:
            data (pd.DataFrame): The dataset from which the labels are to be extracted.

        Returns:
            None
        """
        labels = data["tag"].unique()
        labels = [label.split("_") for label in labels]
        labels = [label for sublist in labels for label in sublist]
        labels = list(set(labels))
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}

    def tokenize_and_align_item(self, example, label_all_tokens=True):
        """
        Tokenizes the input example and aligns the labels with the tokenized input.

        Args:
            example (dict): The input example containing the "morpheme" and "tag" fields.
            label_all_tokens (bool, optional): Whether to include labels for all tokens. Defaults to True.

        Returns:
            dict: The tokenized input with aligned labels.

        """
        tokenized_input = self.tokenizer(example["morpheme"], truncation=True, is_split_into_words=True)
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

    def tokenize(self, label_all_tokens=True):
        """
        Tokenizes the dataset.

        Args:
            tokenizer (XLMRobertaTokenizerFast, optional): The tokenizer to be used for tokenization. Defaults to XLMRobertaTokenizerFast.
            label_all_tokens (bool, optional): Whether to include labels for all tokens. Defaults to True.

        Returns:
            None
        """
        if not self.loaded:
            raise ValueError("Dataset not loaded. Please load the dataset first.")

        if self.tokenized:
            return
        
        self.train = self.train.map(lambda example: self.tokenize_and_align_item(example, label_all_tokens), batched=True)
        self.valid = self.valid.map(lambda example: self.tokenize_and_align_item(example, label_all_tokens), batched=True)
        self.test = self.test.map(lambda example: self.tokenize_and_align_item(example, label_all_tokens), batched=True)

        self.tokenized = True

    def num_labels(self):
        """
        Returns the number of labels in the dataset.

        Returns:
            int: The number of labels in the dataset.
        """
        return len(self.id2label)

    def __len__(self):
        return len(self.train) + len(self.valid) + len(self.test)
    
    def __getitem__(self, idx):
        if idx < len(self.train):
            return self.train[idx]
        elif idx < len(self.train) + len(self.valid):
            return self.valid[idx - len(self.train)]
        else:
            return self.test[idx - len(self.train) - len(self.valid)]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"Dataset({self.language}); Train: {len(self.train)}, Valid: {len(self.valid)}, Test: {len(self.test)}"
    
class MorphParseModel():
    def __init__(self, language: str, path: str="xlm-roberta-large", tokenizer=None, dataset: MorphParseDataset=None, collator=None) -> None:
        self.language = language
        self.path = path
        self.model = None
        self.metrics = evaluate.load("seqeval")
        self.loaded = False

        # loading the tokenizer and dataset
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large", cache_dir=".cache") \
            if tokenizer is None else tokenizer
        self.dataset = MorphParseDataset(language, path="data") if dataset is None else dataset
        self.dataset.load()
        self.dataset.tokenize()
        
        # loading the model and collator
        self.load()
        self.to_gpu()
        self.collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer) if collator is None else collator

        # parsing the arguments for the Trainer
        self.args = self._parse_args()
        self.trainer = self.load_trainer()
    
    def load(self):
        """
        Loads the model from the given path.

        Args:
            path (str, optional): The path to the model. Defaults to "xlm-roberta-large".

        Returns:
            None
        """
        self.model = AutoModelForTokenClassification.from_pretrained(self.path, num_labels=len(self.dataset.id2label), cache_dir=".cache")
        self.loaded = True
    
    def save(self):
        """
        Saves the model to the given path.

        Args:
            path (str, optional): The path to save the model. Defaults to "xlm-roberta-large".

        Returns:
            None
        """
        self.model.config.id2label = self.dataset.id2label
        self.model.config.label2id = self.dataset.label2id
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
    
    def train(self):
        """
        Trains the model.

        Returns:
            None
        """
        self.trainer.train()

    def _compute_metrics(self, eval_preds):
        """
        Compute the metrics for the model evaluation.

        Args:
        eval_preds (tuple): The tuple containing the predictions and labels.

        Returns:
        dict: A dictionary containing the precision, recall, f1 and accuracy scores.
        """

        # get the predictions and labels
        pred_logits, labels = eval_preds
        pred_logits = np.argmax(pred_logits, axis=2)

        # get the predictions and true labels
        # remove the -100 labels from the predictions because they are not real labels but rather morphemes
        # split up into parts by the tokenization

        # adds the mappings to the predictions and true labels with "_-" before the tag since 
        # seqeval requires it
        # 
        predictions = [
            ["_-"+self.dataset.id2label[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_logits, labels)
        ]

        true_labels = [
            ["_-"+self.dataset.id2label[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_logits, labels)
        ]

        results = self.metrics.compute(predictions=predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"] ,
            "accuracy": results["overall_accuracy"],
        }
    def load_trainer(self):
        """
        Returns the Trainer for the model.

        Returns:
            Trainer: The Trainer for the model.
        """
        return Trainer(
            model=self.model,
            args=TrainingArguments(**vars(self.args)),
            train_dataset=self.dataset.train,
            eval_dataset=self.dataset.valid,
            data_collator=self.collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer
        )
    def _parse_args(self):
        """
        Parses the arguments for the Trainer from the command line.

        Returns:
            argparse.Namespace: The parsed arguments.
        """
        parser = argparse.ArgumentParser()
        # core arguments
        parser.add_argument(
            "--output_dir",
            type=str, default="output")
        parser.add_argument(
            "--seed",
            type=int, default=42)
        parser.add_argument(
            "--num_train_epochs",
            type=int, default=3)
        parser.add_argument(
            "--per_device_train_batch_size",
            type=int, default=16)
        parser.add_argument(
            "--per_device_eval_batch_size",
            type=int, default=16)
        parser.add_argument(
            "--learning_rate",
            type=float, default=2e-5)
        parser.add_argument(
            "--weight_decay",
            type=float, default=0.01)
        
        # saving/loading arguments
        parser.add_argument(
            "--evaluation_strategy",
            type=str, default="steps", choices=["steps", "epoch"])
        parser.add_argument(
            "--save_strategy",
            type=str, default="steps", choices=["steps", "epoch"])
        parser.add_argument(
            "--save_steps",
            type=int, default=200)
        parser.add_argument(
            "--eval_steps",
            type=int, default=200)
        parser.add_argument(
            "--save_total_limit",
            type=int, default=2)
            
        parser.add_argument(
            "--load_best_model_at_end",
            action="store_true")
        parser.add_argument(
            "--metric_for_best_model",
            type=str, default="loss")
        parser.add_argument(
            "--greater_is_better",
            action="store_true")
        parser.add_argument(
            "--resume_from_checkpoint",
            action="store_true")
        
        parser.add_argument(
            "--logging_dir",
            type=str, default="logs")
        parser.add_argument(
            "--logging_steps",
            type=int, default=10)
        parser.add_argument(
            "--disable_tqdm",
            action="store_true")
        
        """
        # advanced arguments
        parser.add_argument("--warmup_steps", type=int, default=500)
        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--do_predict", action="store_true")
        parser.add_argument("--overwrite_output_dir", action="store_true")
        parser.add_argument("--overwrite_cache", action="store_true")
        parser.add_argument("--fp16", action="store_true")
        parser.add_argument("--fp16_opt_level", type=str, default="O1")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--max_seq_length", type=int, default=128)
        parser.add_argument("--label_all_tokens", action="store_true")
        parser.add_argument("--logging_first_step", action="store_true")
        parser.add_argument("--eval_steps", type=int, default=10)
        parser.add_argument("--eval_accumulation_steps", type=int, default=1)
        parser.add_argument("--no_cuda", action="store_true")
        parser.add_argument("--ignore_data_skip", action="store_true")
        parser.add_argument("--dataloader_num_workers", type=int, default=0)
        parser.add_argument("--report_to", type=str, default="wandb")
        parser.add_argument("--run_name", type=str, default="run")
        parser.add_argument("--logging_strategy", type=str, default="steps")
        parser.add_argument("--gradient_checkpointing", action="store_true")
        parser.add_argument("--do_predict", action="store_true")
        """

        return parser.parse_known_args()[0]

    def to_gpu(self):
        """
        Moves the model to the GPU.

        Returns:
            None
        """
        if torch.cuda.is_available():
            self.model.to("cuda")

    def __repr__(self):
        return f"MorphParseModel({self.language}) lr={self.args.learning_rate}, epochs={self.args.num_train_epochs}, batch_size={self.args.per_device_train_batch_size}"     
    
    def __str__(self):
        return self.__repr__()
    
class MorphParseArgs():
    """
    Parses the arguments for the MorphParseModel.

    Note: This is a class to allow for easy parsing of the arguments from the command line. Not to be confused with arguments for transformers.TrainingArguments.
    """
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        # core arguments
        self.parser.add_argument(
            "--output_dir",
            type=str, default="output")
        self.parser.add_argument(
            "--model_dir",
            type=str, default="xlm-roberta-large")
        self.parser.add_argument(
            "--data_dir",
            type=str, default="data")
        self.parser.add_argument(
            "--language",
            type=str, default="XH", choices=["NR", "SS", "XH", "ZU"])
        self.parser.add_argument(
            "--cache_dir",
            type=str, default=".cache")
        self.parser.add_argument(
            "--seed",
            type=int, default=42)
        

        self.args = self.parser.parse_known_args()[0]
    
    def __getitem__(self, key):
        return vars(self.args)[key]
    
    def __setitem__(self, key, value):
        vars(self.args)[key] = value

def main():
    # getting the Runner arguments for the program
    args = MorphParseArgs()

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args["model_dir"], cache_dir=args["cache_dir"])
    
    dataset = MorphParseDataset(
        language=args["language"], tokenizer=tokenizer, path=args["data_dir"], seed=args["seed"])
    dataset.load()
    dataset.tokenize()
    
    model = MorphParseModel(
        language=args["language"], path=args["model_dir"], tokenizer=tokenizer, dataset=dataset)
    model.load()

    # grid search for hyperparameters
    # hyperparameters = {
    #     "learning_rate": [1e-5, 2e-5, 3e-5],
    #     "num_train_epochs": [3, 4, 5],
    #     "per_device_train_batch_size": [8, 16, 32]
    # }

    # for lr in hyperparameters["learning_rate"]:
    #     for epochs in hyperparameters["num_train_epochs"]
    #         for batch_size in hyperparameters["per_device_train_batch_size"]:
    #             model.args.learning_rate = lr
    #             model.args.num_train_epochs = epochs
    #             model.args.per_device_train_batch_size = batch_size
    #             model.train()
    #             model.evaluate()
    model.train()

if __name__ == "__main__":
    main()