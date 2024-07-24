import argparse
import json
import pprint
import os
from utils import MorphParseDataset, MorphParseModel
from transformers import XLMRobertaTokenizerFast, pipeline
import torch

def pretrain_models_for_demo(args):
    with open(args.results_file, "r") as f: results = json.load(f)
    res_dict = {}
    top_models = []
    for language in ["NR", "SS", "XH", "ZU"]:
        res_dict[language] = sorted([results[item] for item in results if results[item]["language"]==language], key=lambda x: x["macro_f1"], reverse=True)
        top_models.append(res_dict[language][0])
    
    for config in top_models:
        if os.path.isdir(os.path.join(args.demo_dir, config["language"])):
            continue #pass models that already exist
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(config["model"], cache_dir=args.cache_dir)
        dataset = MorphParseDataset(language=config["language"], path=args.data_dir, tokenizer=tokenizer, validation_split=0)
        dataset.load()
        dataset.tokenize()
        model = MorphParseModel(language=config["language"], path=config["model"], tokenizer=tokenizer, dataset=dataset)
        
        _, learning_rate, batch_size, epoch, _ = config["id"].split("_")
        learning_rate = float(learning_rate) * 1e-5
        batch_size = int(batch_size)
        epoch = int(epoch)

        # training script
        model.args.output_dir = os.path.join(args.demo_dir, config["language"])
        print(model.args.output_dir)
        model.args.learning_rate = learning_rate
        model.args.num_train_epochs = epoch
        model.args.per_device_train_batch_size = batch_size

        model.args.eval_strategy = "no"
        model.args.save_strategy = "steps"
        model.args.save_steps = 1000
        model.args.save_total_limit = 2
        model.args.logging_steps = 1000
        # model.args.load_best_model_at_end = True
        # model.args.metric_for_best_model = "eval_macro-f1"
        # model.args.greater_is_better = True

        model.train()
        model.save()
        
    pass

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

def parse_result(morphs, tags):
    out = ""
    for i, morpheme in enumerate(morphs):
        out += morpheme + f"[{tags[min(i,len(tags))]}]-"
    return out[:-1]

def run_demo(args):
    languages = ["NR", "SS", "XH", "ZU"]

    # loading the pipelines
    if torch.cuda.is_available():
        pipelines = { language: pipeline("ner", model=os.path.join(args.demo_dir, language), tokenizer=XLMRobertaTokenizerFast.from_pretrained(os.path.join(args.demo_dir, language)), device=0, batch_size=16) for language in languages}
    else:
        pipelines = { language: pipeline("ner", model=os.path.join(args.demo_dir, language), tokenizer=XLMRobertaTokenizerFast.from_pretrained(os.path.join(args.demo_dir, language))) for language in languages}

    # loading test datasets
    datasets = { language: MorphParseDataset(language=language, tokenizer=XLMRobertaTokenizerFast.from_pretrained(os.path.join(args.demo_dir, language)), path=args.data_dir, validation_split=0) for language in languages }

    for language in datasets.keys():
        datasets[language].load()
    
    exit_demo = False
    print("at any point you can enter 'exit' to close the application")
    while not exit_demo:
        language = input("enter the language you want to use the parser for:\n('nr', 'ss', 'xh', 'zu')\n")
        if language in ["exit"]:
            break
        parser = pipelines[language.upper()]

        print("enter words with the morphemes space-separated (or 'q' to quit)")

        while True:
            word = input("enter the word:\n")
            if word in ["exit", "q"]:
                exit_demo = word == "exit"
                break

            ner_results = parser(word)
            morphs, tags = format_ner_results(ner_results)
            print("predicted")
            print(parse_result(morphs, tags))
            if word in datasets[language.upper()]:
                print("gold")
                gold = datasets[language.upper()][(datasets[language.upper()].find(word))]
                print(gold["parsed"])




    # ner_results = pipelines["NR"](" ".join(datasets["NR"][0]["morpheme"]))
    # # for item in ner_results: print(item)
    # morphs, tags = format_ner_results(ner_results)

    # print(morphs)
    # print(tags)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="results/config.json")
    parser.add_argument("--demo_dir", default="plm/demo")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default=".cache")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    print(args)
    # pretrain_models_for_demo(args)
    run_demo(args)

if __name__=="__main__":
    main()
