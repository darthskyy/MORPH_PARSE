import warnings
from utils import MorphParseArgs, MorphParseDataset, MorphParseModel
from transformers import XLMRobertaTokenizerFast
import json
import os
import time
import random
import argparse


def create_config_file(filename: str, suffix: str = ""):
    # the grid search configurations.
    # you can add more models, languages, learning rates, batch sizes and epochs to the grid
    GRIDS = {
        "models": ["Davlan/afro-xlmr-large-76L", "xlm-roberta-large", "francois-meyer/nguni-xlmr-large"],
        "languages": ["NR", "SS", "XH", "ZU"],
        "learning_rates": [1e-5, 3e-5, 5e-5],
        "batch_sizes": [16, 32],
        "epochs": [5, 10, 15]
    }

    configs = {}
    for model in GRIDS["models"]:
        for language in GRIDS["languages"]:
            for learning_rate in GRIDS["learning_rates"]:
                for batch_size in GRIDS["batch_sizes"]:
                    for epoch in GRIDS["epochs"]:
                        config = {
                            "id":  f"{language}_{str(learning_rate)[0]}_{batch_size}_{epoch}_{model.split('/')[-1][0]}",
                            "language": language,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "epoch": epoch,
                            "model": model,
                            "completed": False,
                            "macro_f1": 0.0,
                            "micro_f1": 0.0,
                            "macro_recall": 0.0,
                            "micro_recall": 0.0,
                            "macro_precision": 0.0,
                            "micro_precision": 0.0,
                            "loss": 0.0,
                            "runtime": 0.0,
                            "timestamp": 0.0,
                            "suffix": suffix
                        }
                        configs[config["id"]] = config
        with open(filename, 'w') as f:
            json.dump(configs, f, indent=4)

def main():
    # parsing in the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="", choices=["", "_CANON", "_SURFACE"], help="suffix to add to the results file. depends on the type of data we are working with")
    parser.add_argument("--results_file", type=str, default="", help="the file to store the results of the grid search")
    args = parser.parse_args()
    SUFFIX = args.suffix
    RESULTS_FILE = f"results/grid_search_results{SUFFIX}.json" if args.results_file == "" else args.results_file

    # creating the config file if doesn't exist
    # the for loops are nested to create all the possible configurations
    if not os.path.exists(RESULTS_FILE):
        create_config_file(RESULTS_FILE, SUFFIX)
    exit()

    START_TIME = time.time()
    END_TIME = START_TIME + 11.5 * 60 * 60

    # disable the warnings
    warnings.filterwarnings("ignore")

    # LOADING IN THE MORPHPARSE CLASSES
    # getting the Runner arguments for the program
    args = MorphParseArgs()

    experiment_id = f"{args['language']}_{args['model_dir'].split('/')[-1]}" + time.strftime("_%Y%m%d_%H%M%S")
    # loading the tokenizer and the dataset
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args["model_dir"], cache_dir=args["cache_dir"])
    
    dataset = MorphParseDataset(
        language=args["language"], tokenizer=tokenizer, path=args["data_dir"], seed=args["seed"], file_suffix="_SURFACE")
    dataset.load()
    dataset.tokenize()
    
    model = MorphParseModel(
        language=args["language"], path=args["model_dir"], tokenizer=tokenizer, dataset=dataset)
    model.load()

    LANGUAGE = args["language"]
    MODEL_DIR = args["model_dir"]
    

    # MAIN GRID SEARCH LOOP
    rejected_configs = []
    while True:

        # checking for incomplete configurations
        config = json.load(open(RESULTS_FILE))
        incomplete_configs = [id_ for id_ in config if not config[id_]["completed"] and config[id_]["language"] == LANGUAGE and config[id_]["model"] == MODEL_DIR]

        # checks if all the configurations for the Language x Model have been exhausted 
        if len(incomplete_configs) == 0:
            print("all configs are completed")
            exit()

        # checking if we've rejected all the remaining configurations
        if len(rejected_configs) == len(incomplete_configs):
            print("all configs are rejected for remaining time")
            exit()

        # picks a random configuration to work on which hasn't been rejected yet because of time constraints
        curr_config = random.choice([x for x in incomplete_configs if x not in rejected_configs])
        
        print(f"Selected: {curr_config}")
        config[curr_config]["completed"] = "running"
        # writes to the file that it's running to avoid doubly running configurations
        with open(RESULTS_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        
        # extracts the hyperparameters
        _, learning_rate, batch_size, epoch, _ = curr_config.split("_")
        learning_rate = float(learning_rate) * 1e-5
        batch_size = int(batch_size)
        epoch = int(epoch)

        # checking for time constraints
        # you can comment out the following if statements to run the script without time constraints
        if (epoch == 15 and batch_size == 8 and END_TIME - time.time() < 3.5*60*60) or (epoch == 15 and END_TIME - time.time() < 2*60*60):
            config[curr_config]["completed"] = False
            with open(RESULTS_FILE, 'w') as f:
                json.dump(config, f, indent=4)

            rejected_configs.append(curr_config)
            continue

        if END_TIME - time.time() < 0.3*60*60:
            config[curr_config]["completed"] = False
            with open(RESULTS_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            break


        # TRAINING THE MODEL STARTS HERE
        # setting the hyperparameters
        model.args.learning_rate = learning_rate
        model.args.num_train_epochs = epoch
        model.args.per_device_train_batch_size = batch_size
        
        print(f"running: {curr_config}")
        start = time.time()
        model.train()
        results = model.evaluate()
        runtime = float(time.time() - start)/3600
        print(f"{curr_config} final results")


        # reading the config file again to update the results
        del config
        config = json.load(open(RESULTS_FILE, 'r'))

        print(results)
        config[curr_config]["completed"] = True
        config[curr_config]["macro_f1"] = results["eval_macro-f1"]
        config[curr_config]["micro_f1"] = results["eval_f1"]
        config[curr_config]["macro_recall"] = results["eval_macro-recall"]
        config[curr_config]["micro_recall"] = results["eval_recall"]
        config[curr_config]["macro_precision"] = results["eval_macro-precision"]
        config[curr_config]["micro_precision"] = results["eval_precision"]
        config[curr_config]["loss"] = results["eval_loss"]
        config[curr_config]["runtime"] = runtime
        config[curr_config]["timestamp"] = time.time()

        with open(RESULTS_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        os.system(f"rm {model.args.output_dir}/* -rf")
        print(f"Finished: {curr_config}")
    print("Script done")
if __name__ == "__main__":
    main()