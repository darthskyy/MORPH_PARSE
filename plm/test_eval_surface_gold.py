from utils import MorphParseArgs, MorphParseDataset, MorphParseModel, GenUtils
from transformers import XLMRobertaTokenizerFast, pipeline

from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

import torch
from datasets import Dataset
import json
import os
import pprint
import time
import pandas as pd
import csv
import copy
import random
import itertools

import warnings
warnings.filterwarnings("ignore")
START_TIME = time.time()
END_TIME = START_TIME + 11.5 * 60 * 60

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

args = MorphParseArgs()


# variables for the output files
SUFFIX = "_SURFACE_GOLD"
RESULTS_FILE = f"results/final{SUFFIX}.json"

# load the config files to get the best models for each of the languages
LANGUAGES = ["NR", "SS", "XH", "ZU"]
MODELS = ["Davlan/afro-xlmr-large-76L", "xlm-roberta-large", "francois-meyer/nguni-xlmr-large"]
CONFIG_FILE = "results/config.json"

config = json.load(open(CONFIG_FILE, "r"))

# get the best models for each language-model pair
pairs = list(itertools.product(LANGUAGES, MODELS))
best_models = {}
best_models = {
    pair: sorted([copy.deepcopy(config[item]) for item in config if config[item]["language"] == pair[0] and config[item]["model"] == pair[1]], key=lambda x: x["macro_f1"], reverse=True)[0] for pair in pairs
}

# for i, pair in enumerate(best_models):
#     print(i,pair)


# deleting the original config dictionary because we've already gotten what we need from it
del config


# get the results from the gold to see which one had the best test performance
# config = json.load(open("results/final.json", "r"))

# results = {}
# for pair in pairs:
#     seed = sorted([copy.deepcopy(config[item]) for item in config if config[item]["language"] == pair[0] and config[item]["model"] == pair[1]], key=lambda x: x["macro_f1"], reverse=True)[0]
#     pprint.pprint(seeds)
#     model_id = seed["id"]
#     results[model_id]["suffix"] = SUFFIX
#     results[model_id]["completed"] = False
#     results[model_id]["timestamp"] = 0
#     results[model_id]["runtime"] = 0
#     results[model_id]["micro_f1"] = 0
#     results[model_id]["macro_f1"] = 0
#     results[model_id]["micro_recall"] = 0
#     results[model_id]["macro_recall"] = 0
#     results[model_id]["micro_precision"] = 0
#     results[model_id]["macro_precision"] = 0
#     results[model_id]["loss"] = 0
#     results[model_id]["mismatches"] = 0

# if not os.path.exists(RESULTS_FILE):
#     config = json.load(open("results/final.json", "r"))

#     results = {}
#     for pair in pairs:
#         seed = sorted([copy.deepcopy(config[item]) for item in config if config[item]["language"] == pair[0] and config[item]["model"] == pair[1]], key=lambda x: x["macro_f1"], reverse=True)[0]
#         # pprint.pprint(seed)
#         model_id = seed["id"]
#         results[model_id] = seed
#         results[model_id]["suffix"] = SUFFIX
#         results[model_id]["completed"] = False
#         results[model_id]["timestamp"] = 0
#         results[model_id]["runtime"] = 0
#         results[model_id]["micro_f1"] = 0
#         results[model_id]["macro_f1"] = 0
#         results[model_id]["micro_recall"] = 0
#         results[model_id]["macro_recall"] = 0
#         results[model_id]["micro_precision"] = 0
#         results[model_id]["macro_precision"] = 0
#         results[model_id]["loss"] = 0
#         results[model_id]["mismatches"] = 0
    
#     with open(RESULTS_FILE, "w") as f:
#         json.dump(results, f, indent=4)

#     del config

if not os.path.exists(RESULTS_FILE):
    results = {}
    for language in LANGUAGES:
        for model in MODELS:
            for seed in range(1, 6):
                model_id = f"{language}_{model.split('/')[-1][0]}_{seed}"
                results[model_id] = copy.deepcopy(best_models[(language, model)])
                results[model_id]["id"] = model_id
                results[model_id]["seed"] = seed
                results[model_id]["suffix"] = SUFFIX
                results[model_id]["completed"] = False
                results[model_id]["timestamp"] = 0
                results[model_id]["runtime"] = 0
                results[model_id]["micro_f1"] = 0
                results[model_id]["macro_f1"] = 0
                results[model_id]["micro_recall"] = 0
                results[model_id]["macro_recall"] = 0
                results[model_id]["micro_precision"] = 0
                results[model_id]["macro_precision"] = 0
                results[model_id]["loss"] = 0
                results[model_id]["mismatches"] = 0
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)


# to count how many models were completed
model_count = 0

# check the bad configurations
with open(f"results/bad{SUFFIX}.txt", "r") as f:
    bad_config = [line.strip() for line in f.readlines()]

while True:
    # load the final results file
    config = json.load(open(RESULTS_FILE, "r"))
    remaining_models_1 = [item for item in config if not config[item]["completed"] and config[item]["seed"] in ["1", 1] and item not in bad_config]
    remaining_models_2 = [item for item in config if not config[item]["completed"] and config[item]["seed"] in ["2", 2] and item not in bad_config]
    remaining_models_3 = [item for item in config if not config[item]["completed"] and config[item]["seed"] in ["3", 3] and item not in bad_config]
    remaining_models_4 = [item for item in config if not config[item]["completed"] and config[item]["seed"] in ["4", 4] and item not in bad_config]
    remaining_models_5 = [item for item in config if not config[item]["completed"] and config[item]["seed"] in ["5", 5] and item not in bad_config]
    
    # premature stopping of the script if there is little time left for training
    if END_TIME - time.time() < 0.3*60*60:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print("Time is up!")
        print(f"trained {model_count} models in {(END_TIME-time.time())/3600} hours.")
        exit()

    start = time.time()
    # getting one seed at a time done
    # if there are no more models to train, exit
    if remaining_models_1:
        curr_config = random.choice(remaining_models_1)
    elif remaining_models_2:
        curr_config = random.choice(remaining_models_2)
    elif remaining_models_3:
        curr_config = random.choice(remaining_models_3)
    elif remaining_models_4:
        curr_config = random.choice(remaining_models_4)
    elif remaining_models_5:
        curr_config = random.choice(remaining_models_5)
    else:
        print(f"All models have been trained.")
        exit()


    print(f"Starting with {curr_config}")
    config[curr_config]["completed"] = "running"
    # writes to the file that it's running to avoid doubly running configurations
    # this is important because the script is run in parallel
    with open(RESULTS_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    # regular data loading and training
    language = config[curr_config]["language"]
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(config[curr_config]["model"], cache_dir=args["cache_dir"])

    dataset = MorphParseDataset(language=language, path=args["data_dir"], tokenizer=tokenizer, validation_split=0, file_suffix=SUFFIX)
    dataset.load()
    dataset.tokenize()

    model = MorphParseModel(language=language, path=config[curr_config]["model"], tokenizer=tokenizer, dataset=dataset)
    model.load()

    model.args.learning_rate = float(config[curr_config]["learning_rate"])
    model.args.per_device_train_batch_size = int(config[curr_config]["batch_size"])
    model.args.num_train_epochs = int(config[curr_config]["epoch"])
    model.args.seed = int(config[curr_config]["seed"])
    model.args.output_dir = os.path.join("tests", language, f"seed_{config[curr_config]['seed']}")
    model.args.save_strategy = "epoch"
    model.args.save_steps = 0
    model.args.save_total_limit = 1
    model.args.logging_steps = 1000
    model.args.eval_strategy = "no"

    model.train()


    # evaluation using the NER pipeline
    if torch.cuda.is_available():
        nlp = pipeline("ner", model=model.model, tokenizer=dataset.tokenizer, device=0, batch_size=16)
    else:
        nlp = pipeline("ner", model=model.model, tokenizer=dataset.tokenizer)
    print("Pipeline created")


    predictions = []
    references = []
    lines = ["morphemes\ttarget\tprediction\n"]
    test_set = dataset.test
    mismatches = 0
    for i in range(len(test_set)):
        sentence = " ".join(test_set[i]["morpheme"])
        ner_results = nlp(sentence)
        morphs, tags = format_ner_results(ner_results)
        expected_tags = ["_-" + dataset.id2label[x] for x in test_set[i]["tag"]]
        tags = ["_-" + x for x in tags]
        if len(tags) != len(expected_tags):
            mismatches += 1
        tags, expected_tags = GenUtils.align_seqs(tags, expected_tags)
        output = ""
        output += "_".join([trg[2:] for trg in morphs]) + "\t"
        output += "_".join([trg[2:] for trg in expected_tags]) + "\t"
        output += "_".join([trg[2:] for trg in tags]) + "\n"
        lines.append(output)

        if len(expected_tags) != len(tags):
            continue
        predictions.append(tags)
        references.append(expected_tags)
    
    with open(f"predictions/{SUFFIX}/{curr_config}.tsv", "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    
    results = classification_report(references, predictions, output_dict=True)

    results = {
        "micro_precision": results["micro avg"]["precision"],
        "micro_recall": results["micro avg"]["recall"],
        "micro_f1": results["micro avg"]["f1-score"] ,
        "macro_f1": results["macro avg"]["f1-score"],
        "macro_precision": results["macro avg"]["precision"],
        "macro_recall": results["macro avg"]["recall"],
    }

    del config
    # get config again to make sure you're up to date before we update
    config = json.load(open(RESULTS_FILE, "r"))

    # skipping bad configurations
    if results["micro_f1"] < 0.10:
        with open(f"results/bad{SUFFIX}.txt", "a") as f:
            print(curr_config, file=f)
        bad_config.append(curr_config)
        config[curr_config]["completed"] = False
        with open(RESULTS_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Skipping {curr_config} due to low f1 score")
        os.system(f"rm {model.args.output_dir}/* -rf")
        del model
        del dataset
        del tokenizer
        del nlp
        del test_set
        del predictions
        del references
        del config
        torch.cuda.empty_cache()
        time.sleep(10)
        continue

    # write the results to the file
    config[curr_config]["micro_f1"] = results["micro_f1"]
    config[curr_config]["micro_precision"] = results["micro_precision"]
    config[curr_config]["micro_recall"] = results["micro_recall"]
    config[curr_config]["macro_f1"] = results["macro_f1"]
    config[curr_config]["macro_precision"] = results["macro_precision"]
    config[curr_config]["macro_recall"] = results["macro_recall"]
    config[curr_config]["completed"] = True
    config[curr_config]["timestamp"] = time.time()
    config[curr_config]["mismatches"] = mismatches
    config[curr_config]["runtime"] = float(time.time() - start)/3600

    with open(RESULTS_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    model_count+=1
    print(f"{curr_config} final results")
    print(results)
    print(f"Runtime: {config[curr_config]['runtime']} hours")
    print(f"Timestamp: {config[curr_config]['timestamp']}")
    print("-------------------------------------------------")
    os.system(f"rm {model.args.output_dir}/* -rf")
    del model
    del dataset
    del tokenizer
    del nlp
    del test_set
    del predictions
    del references
    del config

    torch.cuda.empty_cache()
    time.sleep(10)
