import pandas as pd
import warnings
import argparse
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, pipeline, BertTokenizerFast

# creating a parser for the command line arguments
parser = argparse.ArgumentParser(description="Test the fine-tuned model on the test set")
parser.add_argument("--model", type=str, help="The path to the fine-tuned model", required=True)
parser.add_argument("--tokenizer", type=str, help="The path to the tokenizer", required=True)
parser.add_argument("--test", type=str, help="The path to the test set", required=True)
parser.add_argument("--output", type=bool, help="The path to the output file", default=False)
parser.add_argument("--metric", type=str, help="The metric to use for evaluation", default="f1")
args = parser.parse_args()

# loading the model and tokenizer
print("Loading model...", end="\r")
model_fine_tuned = AutoModelForTokenClassification.from_pretrained(args.model)
print("Model loaded____", end="\r")

print("Loading tokenizer...", end="\r")
tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
print("Tokenizer loaded____", end="\r")

print("Creating NLP pipeline...", end="\r")
nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)
print("NLP pipeline created____", end="\r")


# loading the test set
columns = ["x", "y", "morphs", "tags"]
test_df = pd.read_csv(args.test, delimiter="\t", names=columns)
print("Test set loaded_______________")


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

            if morph.startswith("_"):
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


print("Testing__________________________", end="\r")
references = []
predictions = []
warnings.filterwarnings("ignore")

# iterating over the test set and making predictions'
print("                                  ", end="\r")
print(f"    /{len(test_df)}", end="\r")
for i, item in test_df.iterrows():
    example = item["morphs"].replace("_", " ")
    expected_tags = item["tags"].split("_")
    ner_results = nlp(example)
    morphs, tags = format_ner_results(ner_results)
    expected_tags = [t for t in expected_tags if "Dem" not in t]
    expected_tags = ["_-" + t for t in expected_tags]
    tags = ["_-" + t for t in tags]

    # if the number of tags is not the same, skip the example
    if len(expected_tags) != len(tags):
        continue

    references.append(expected_tags)
    predictions.append(tags)
    print(f"{i + 1:>4}", end="\r")

print("Evaluation")

if args.metric == "f1":
    print(f"F1 score: {f1_score(references, predictions)}")
elif args.metric == "precision":
    print(f"Precision score: {precision_score(references, predictions)}")
elif args.metric == "recall":
    print(f"Recall score: {recall_score(references, predictions)}")
else:
    print(classification_report(references, predictions))

print(f"Number of examples: {len(references)}")
print("Done")