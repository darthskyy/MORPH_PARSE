from utils import MorphParseArgs, MorphParseDataset, MorphParseModel, GenUtils
from transformers import XLMRobertaTokenizerFast, pipeline

from seqeval.metrics import classification_report

import argparse
import torch
import json
import os
import time
import copy
import random
import itertools

import warnings
warnings.filterwarnings("ignore")

def create_results_file(results_file: str, suffix: str, grid_search_file: str="results/config.json"):
    # load the config files to get the best models for each of the languages
    LANGUAGES = ["NR", "SS", "XH", "ZU"]
    MODELS = ["Davlan/afro-xlmr-large-76L", "xlm-roberta-large", "francois-meyer/nguni-xlmr-large"]
    CONFIG_FILE = grid_search_file

    config = json.load(open(CONFIG_FILE, "r"))

    # get the best models for each language-model pair
    pairs = list(itertools.product(LANGUAGES, MODELS))
    best_models = {}
    best_models = {
        pair: sorted([copy.deepcopy(config[item]) for item in config if config[item]["language"] == pair[0] and config[item]["model"] == pair[1]], key=lambda x: x["macro_f1"], reverse=True)[0] for pair in pairs
    }

    # deleting the original config dictionary because we've already gotten what we need from it
    del config

    results = {}
    for language in LANGUAGES:
        for model in MODELS:
            for seed in range(1, 6):
                model_id = f"{language}_{model.split('/')[-1][0]}_{seed}"
                results[model_id] = copy.deepcopy(best_models[(language, model)])
                results[model_id]["id"] = model_id
                results[model_id]["seed"] = seed
                results[model_id]["suffix"] = suffix
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
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

def main():
    START_TIME = time.time()
    END_TIME = START_TIME + 11.5 * 60 * 60

    # parsing in the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="", choices=["", "_CANON", "_SURFACE", "_SURFACE_GOLD"], help="suffix to add to the results file. depends on the type of data we are working with")
    parser.add_argument("--results_file", type=str, default="", help="the file to store the results of the grid search")
    parser.add_argument("--grid_search_file", type=str, default="results/config.json", help="the file to store the results of the grid search")
    args = parser.parse_args()
    # variables for the output files
    SUFFIX = args.suffix
    RESULTS_FILE = f"results/final_results{SUFFIX}.json" if args.results_file == "" else args.results_file
    GRID_SEARCH_FILE = args.grid_search_file

    # creating the results file if it doesn't exist
    if not os.path.exists(RESULTS_FILE):
        create_results_file(RESULTS_FILE, SUFFIX, GRID_SEARCH_FILE)

    args = MorphParseArgs()
    # to count how many models were completed
    model_count = 0

    # check the bad configurations
    if not os.path.exists(f"results/bad{SUFFIX}.txt"):
        with open(f"results/bad{SUFFIX}.txt", "w") as f:
            pass
    
    with open(f"results/bad{SUFFIX}.txt", "r") as f:
        bad_config = [line.strip() for line in f.readlines()]

    while True:
        # load the final results file
        config = json.load(open(RESULTS_FILE, "r"))
        # get the remaining models to train
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
        
        # if there are no more models to train, exit
        # getting one seed at a time done
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

        ## TRAINING THE MODEL
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
            parser = pipeline("ner", model=model.model, tokenizer=dataset.tokenizer, device=0, batch_size=16)
        else:
            parser = pipeline("ner", model=model.model, tokenizer=dataset.tokenizer)
        print("Pipeline created")


        def reverse_ids(ids):
            return [dataset.id2label[id] for id in ids]

        # getting the morphemes, references, and predictions
        # morphemes are space separated so that the parser can process them
        morphemes = dataset.test.to_pandas()['morpheme'].apply(lambda x: ' '.join(x)).tolist()
        # references are converted from ids to labels and the Pos tags are removed as it often exists in a double tag
        references = dataset.test.to_pandas()['tag'].apply(lambda x: reverse_ids(x)).tolist()
        # predictions are obtained from the parser and converted to labels
        predictions = parser(morphemes)
        predictions = [GenUtils.format_ner_results(p)[1] for p in predictions]
        
        # regetting the morphemes
        morphemes = dataset.test.to_pandas()['morpheme'].apply(lambda x: '_'.join(x)).tolist()
        lines = ["morphemes\ttarget\tprediction\n"]
        mismatches = 0
        for i in range(len(predictions)):
            if len(predictions[i]) != len(references[i]):
                mismatches += 1
            predictions[i], references[i] = GenUtils.align_seqs(predictions[i], references[i])
            
            # write the morphemes, references, and predictions to a file
            output = ""
            output += morphemes[i] + "\t"
            output += "_".join(references[i]) + "\t"
            output += "_".join(predictions[i]) + "\n"
            lines.append(output)

            # add the # to the predictions and references because the classification report expects NER
            predictions[i] = ["#" + p for p in predictions[i]]
            references[i] = ["#" + r for r in references[i]]

        
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
            del parser
            del test_set
            del predictions
            del references
            del config
            torch.cuda.empty_cache()
            time.sleep(10)
            continue
        
        # writing the predictions to a file
        if not os.path.exists(f"plm/predictions"):
            os.mkdir("plm/predictions")
        if SUFFIX == "":
            SUFFIX = "PLAIN"
        if not os.path.exists(f"plm/predictions/{SUFFIX}"):
            os.mkdir(f"plm/predictions/{SUFFIX}")
        with open(f"plm/predictions/{SUFFIX}/{curr_config}.tsv", "w", encoding="utf-8") as f:
            f.writelines(lines)
        SUFFIX = "" if SUFFIX=="PLAIN" else SUFFIX
        
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
        del parser
        del test_set
        del predictions
        del references
        del config

        torch.cuda.empty_cache()
        time.sleep(10)

if __name__ == "__main__":
    main()