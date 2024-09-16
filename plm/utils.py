import argparse
from datetime import datetime
import os
import csv
import re
import warnings


import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report
from transformers import XLMRobertaTokenizerFast, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments


class MorphParseDataset():
    def __init__(self, language: str, path: str="data", validation_split: float=0.1, seed: int=42, tokenizer=None, file_suffix: str="") -> None:
        self.language = language
        self.path = path
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large", cache_dir=".cache") \
            if tokenizer is None else tokenizer
        self.train = None
        self.valid = None
        self.test = None
        self.file_suffix = file_suffix

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

        train_path = os.path.join(self.path, "TRAIN", f"{self.language}_TRAIN{self.file_suffix}.tsv")
        test_path = os.path.join(self.path, "TEST", f"{self.language}_TEST{self.file_suffix}.tsv")
        # valid_path = os.path.join(path, "VALID", f"{self.language}_VALID{self.file_suffix}.tsv")

        train_data = pd.read_csv(train_path, delimiter="\t", names=self.column_names, quoting=csv.QUOTE_NONE)
        test_data = pd.read_csv(test_path, delimiter="\t", names=self.column_names, quoting=csv.QUOTE_NONE)

        # uncomment this if you want to remove the double tags [Dem\d]_[Pos\d]
        # pos_pattern = re.compile(r'Pos\d')
        # def remove_pos(tags):
        #     return [tag for tag in tags if not pos_pattern.match(tag)]
        # train_data["tag"] = train_data["tag"].apply(lambda x: x.split("_")).apply(remove_pos).apply(lambda x: "_".join(x))
        # test_data["tag"] = test_data["tag"].apply(lambda x: x.split("_")).apply(remove_pos).apply(lambda x: "_".join(x))

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
        # self.test = self.test.map(lambda example: self.tokenize_and_align_item(example, label_all_tokens), batched=True)

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
    
    def __contains__(self, key):
        for item in self:
            if key.lower() in [item["word"].lower(), item["parsed"].lower(), " ".join(item["morpheme"]), "_".join(item["morpheme"])]:
                return True
        return False
    
    def find(self, key):
        for idx, item in enumerate(self):
            if key.lower() in [item["word"].lower(), item["parsed"].lower(), " ".join(item["morpheme"]), "_".join(item["morpheme"])]:
                return idx
        return -1

    def __repr__(self):
        return f"Dataset({self.language}); Train: {len(self.train)}, Valid: {len(self.valid)}, Test: {len(self.test)}"
    
class MorphParseModel():
    def __init__(self, language: str, path: str="xlm-roberta-large", tokenizer=None, dataset: MorphParseDataset=None, collator=None) -> None:
        self.language = language
        self.path = path
        self.model = None
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
        self.trainer = None
        self.load_trainer()
    
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
    
    def get_model(self):
        return self.model
    
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
        self.load_trainer()
        self.trainer.train()
        self.model.config.id2label = self.dataset.id2label
        self.model.config.label2id = self.dataset.label2id

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

        results = classification_report(true_labels, predictions, output_dict=True)

        return {
            "precision": results["micro avg"]["precision"],
            "recall": results["micro avg"]["recall"],
            "f1": results["micro avg"]["f1-score"] ,
            "macro-f1": results["macro avg"]["f1-score"],
            "macro-precision": results["macro avg"]["precision"],
            "macro-recall": results["macro avg"]["recall"],
        }
    
    def load_trainer(self):
        """
        Returns the Trainer for the model.

        Returns:
            Trainer: The Trainer for the model.
        """
        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**vars(self.args)),
            train_dataset=self.dataset.train,
            eval_dataset=self.dataset.valid,
            data_collator=self.collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer
        )
    
    def get_trainer(self):
        self.load_trainer()
        return self.trainer
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
            "--eval_strategy",
            type=str, default="steps", choices=["no", "steps", "epoch"])
        parser.add_argument(
            "--save_strategy",
            type=str, default="steps", choices=["no", "steps", "epoch"])
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
            "--logging_steps",
            type=int, default=100)
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
            torch.cuda.empty_cache()
            self.model.to("cuda")
            print("gpu available")
        else:
            print("gpu not available")

    def evaluate(self):
        """
        Evaluates the model.

        Returns:
            dict: A dictionary containing the precision, recall, f1 and accuracy scores.
        """
        return self.trainer.evaluate()
    
    def evaluate_test(self):
        """
        Evaluates the model on the test set.

        Returns:
            dict: A dictionary containing the precision, recall, f1 and accuracy scores.
        """
        return self.trainer.evaluate(self.dataset.test)
    
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
            type=str, default="output",
            help="The output directory where the model predictions and checkpoints will be written.")
        self.parser.add_argument(
            "--model_dir",
            type=str, default="xlm-roberta-large",
            help="The directory or model identifier from huggingface.co/models where the model is loaded from.")
        self.parser.add_argument(
            "--data_dir",
            type=str, default="data",
            help="The input data directory. Should contain the training files for the NER task.")
        self.parser.add_argument(
            "--log_file",
            type=str, default="logs.log",
            help="The file where the training and evaluation logs will be written.")
        self.parser.add_argument(
            "--language",
            type=str, default="XH", choices=["NR", "SS", "XH", "ZU"],
            help="The language code for the dataset to be used. Choices are: 'NR', 'SS', 'XH', 'ZU'.")
        self.parser.add_argument(
            "--cache_dir",
            type=str, default=".cache",
            help="The directory where the pretrained models downloaded from huggingface.co will be cached.")
        self.parser.add_argument(
            "--seed",
            type=int, default=42,
            help="Random seed for initialization.")
        
        # for the model (repeated here for the help function)
        self.parser.add_argument(
            "--num_train_epochs",
            type=int, default=3,
            help="Total number of training epochs to perform.")
        self.parser.add_argument(
            "--per_device_train_batch_size",
            type=int, default=16,
            help="Batch size per GPU/TPU core/CPU for training.")
        self.parser.add_argument(
            "--per_device_eval_batch_size",
            type=int, default=16,
            help="Batch size per GPU/TPU core/CPU for evaluation.")
        self.parser.add_argument(
            "--learning_rate",
            type=float, default=2e-5,
            help="The initial learning rate for Adam.")
        self.parser.add_argument(
            "--weight_decay",
            type=float, default=0.01,
            help="Weight decay if we apply some.")
        
        # saving/loading arguments
        self.parser.add_argument(
            "--eval_strategy",
            type=str, default="steps", choices=["no", "steps", "epoch"],
            help="The evaluation strategy to use.")
        self.parser.add_argument(
            "--eval_steps",
            type=int, default=200,
            help="Number of update steps between two evaluations. Can only be used if eval_strategy set to 'steps'.")
        self.parser.add_argument(
            "--save_strategy",
            type=str, default="steps", choices=["no", "steps", "epoch"],
            help="The checkpoint save strategy to use.")
        self.parser.add_argument(
            "--save_steps",
            type=int, default=200,
            help="Number of updates steps before two checkpoint saves.")
        self.parser.add_argument(
            "--save_total_limit",
            type=int, default=2,
            help="Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir.")
            
        self.parser.add_argument(
            "--load_best_model_at_end",
            action="store_true",
            help="Whether to load the best model found at the end of training.")
        self.parser.add_argument(
            "--metric_for_best_model",
            type=str, default="loss",
            help="The metric to use to compare the best model.")
        self.parser.add_argument(
            "--greater_is_better",
            action="store_true",
            help="Whether the `metric_for_best_model` should be maximized or not.")
        self.parser.add_argument(
            "--resume_from_checkpoint",
            action="store_true",
            help="Whether to resume training from the last checkpoint.")
        
        self.parser.add_argument(
            "--logging_steps",
            type=int, default=100,
            help="Number of update steps between two logs.")
        self.parser.add_argument(
            "--disable_tqdm",
            action="store_true",
            help="Whether to disable the tqdm progress bars.")
        
        
        self.parser.add_argument("--predictions", type=str, default="", help="The file to save the predictions")
        self.parser.add_argument("--train", type=bool, default=True, help="Whether to train the model")
        

        self.args = self.parser.parse_known_args()[0]
    

    def __getattr__(self, key):
        return getattr(self.args, key)
    
    def __setattr__(self, key, value):
        if key == "parser" or key == "args":
            super().__setattr__(key, value)
        else:
            setattr(self.args, key, value)

    def __getitem__(self, key):
        return vars(self.args)[key]
    
    def __setitem__(self, key, value):
        vars(self.args)[key] = value

class GenUtils():
    def align_seqs(pred: list, actual: list, pad="<?UNK?>"):
        """
        Align two sequences (`pred` and `actual`) by inserting `pad`ding elements into the shorter sequence such that we
        maximise the number of matches between the two lists. Elements are never swapped in order or transmuted.

        Examples
        ========

        A shorter predicted sequence will have a padding element inserted:
        >>> print(align_seqs(["Hey", "there", "neighbour!"], ["Hey", "there", "my", "neighbour!"]))
            (['Hey', 'there', '<?UNK?>', 'neighbour!'], ['Hey', 'there', 'my', 'neighbour!'])

        A shorter actual sequence will have a padding element inserted:
        >>> print(align_seqs(["Hey", "there", "neighbour!"], ["Hey", "there"]))
            (['Hey', 'there', 'neighbour!'], ['Hey', 'there', '<?UNK?>'])

        Sequences of equal length do not change:
        >>> print(align_seqs(["Hey", "hello", "there", "neighbour!"], ["Hey", "there", "my", "neighbour!"]))
            (['Hey', 'hello', 'there', 'neighbour!'], ['Hey', 'there', 'my', 'neighbour!'])
        """
        longer, shorter = (pred, actual) if len(pred) > len(actual) else (actual, pred)

        best_correct = 0
        best_indices = []

        for indices in GenUtils._gen_indices(len(longer) - len(shorter), len(longer) - 1):
            # Just copy the shorter one as it's what we will insert into
            pred_copy, actual_copy = (pred, GenUtils._copy_and_insert(actual, indices, pad)) if len(pred) > len(actual) else (
                GenUtils._copy_and_insert(pred, indices, pad), actual)

            correct = 0
            for (pred_elt, actual_elt) in zip(pred_copy, actual_copy):
                if pred_elt == actual_elt:
                    correct += 1
        
            if correct >= best_correct:
                best_indices = indices
                best_correct = correct

            if best_correct == len(shorter):
                break

        shorter = GenUtils._copy_and_insert(shorter, best_indices, pad)

        return (longer, shorter) if len(pred) > len(actual) else (shorter, longer)

    def _copy_and_insert(seq: list, indices, pad):
        """Copy the given `seq` and insert the `pad` element at the specified `indices`, returning the post-insert list"""
        copied = seq.copy()

        for index in indices:
            copied.insert(index, pad)

        return copied


    def _gen_indices(length_diff: int, max_idx: int) -> list:
        """
        Generate all possible sets of indices that we can insert elements into the smaller list in order for it to
        become as long as the longest list. In this function, we are not concerned with which are the _optimal_ indices -
        we just generate all of them and filter later.
        """

        # Let's say we have shortest_len = 7 and longest = 10
        # We need to find (x, y, z) s.t. x <= y <= z
        # So, first pick x in the range 10..=0
        # Then we pick y in the range x..=0
        # Then z in the range y..=0
        # etc

        if length_diff == 0:
            return []

        for first_coord in range(max_idx, -1, -1):
            if length_diff == 1:
                yield (first_coord,)
            else:
                for indices in GenUtils._gen_indices(length_diff - 1, first_coord):
                    yield first_coord, *indices

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

def main():
    # disable the warnings
    warnings.filterwarnings("ignore")
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
    hyperparameters = {
        "learning_rate": [1e-5, 2e-5, 3e-5],
        "num_train_epochs": [3, 5, 10],
        "per_device_train_batch_size": [8, 16, 32]
    }

    for epochs in hyperparameters["num_train_epochs"]:
        for lr in hyperparameters["learning_rate"]:
            for batch_size in hyperparameters["per_device_train_batch_size"]:
                model.args.learning_rate = lr
                model.args.num_train_epochs = epochs
                model.args.per_device_train_batch_size = batch_size
                print(model)
                model.train()
                results = model.evaluate()
                with open(args["log_file"], "a") as f:
                    f.write(f"Results for {model}: {results}\n")
    # model.train()

if __name__ == "__main__":
    main()