import warnings
from utils import MorphParseArgs, MorphParseDataset, MorphParseModel
from transformers import XLMRobertaTokenizerFast
import json
import os
import pprint
import time
import random
import sys

def main():
    START_TIME = time.time()
    END_TIME = START_TIME + 11.5 * 60 * 60

    # disable the warnings
    warnings.filterwarnings("ignore")
    # getting the Runner arguments for the program
    args = MorphParseArgs()

    experiment_id = f"{args['language']}_{args['model_dir'].split('/')[-1]}" + time.strftime("_%Y%m%d_%H%M%S")
    # loading the tokenizer and the dataset
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args["model_dir"], cache_dir=args["cache_dir"])
    
    dataset = MorphParseDataset(
        language=args["language"], tokenizer=tokenizer, path=args["data_dir"], seed=args["seed"])
    dataset.load()
    dataset.tokenize()
    
    model = MorphParseModel(
        language=args["language"], path=args["model_dir"], tokenizer=tokenizer, dataset=dataset)
    model.load()

    # creating the config file if doesn't exist
    if not os.path.exists('results/config.json'):
        configs = {}
        for model in ["Davlan/afro-xlmr-large-76L", "xlm-roberta-large"]:
            for language in ["NR", "SS", "XH", "ZU"]:
                for learning_rate in [1e-5, 3e-5, 5e-5]:
                    for batch_size in [16, 32]:
                        for epoch in [5, 10, 15]:
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
                                "timestamp": 0.0
                            }
                            configs[config["id"]] = config
        with open('results/config.json', 'w') as f:
            json.dump(configs, f, indent=4)

    lang = args["language"]
    model_name = args["model_dir"]
    
    rejected_configs = []
    while True:

        # checking for incomplete configurations
        config = json.load(open('results/config.json'))
        incomplete_configs = [id_ for id_ in config if not config[id_]["completed"] and config[id_]["language"] == lang and config[id_]["model"] == model_name]

        # checks if all the configurations for the Language x Model have been exhausted 
        if len(incomplete_configs) == 0:
            print("all configs are completed")
            sys.exit()

        # checking if we've rejected all the remaining configurations
        if len(rejected_configs) == len(incomplete_configs):
            print("all configs are rejected for remaining time")
            sys.exit()

        # picks a random configuration to work on which hasn't been rejected yet because of time constraints
        curr_config = random.choice([x for x in incomplete_configs if x not in rejected_configs])

        # skipping over 8 batch size for now
        if "_8_" in curr_config:
            rejected_configs.append(curr_config)
            continue
        
        print(f"Selected: {curr_config}")
        config[curr_config]["completed"] = "running"
        # writes to the file that it's running to avoid doubly running configurations
        with open('results/config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        # extracts the hyperparameters
        _, learning_rate, batch_size, epoch, _ = curr_config.split("_")
        learning_rate = float(learning_rate) * 1e-5
        batch_size = int(batch_size)
        epoch = int(epoch)

        # checking for time constraints
        if (epoch == 15 and batch_size == 8 and END_TIME - time.time() < 3.5*60*60) or (epoch == 15 and END_TIME - time.time() < 2*60*60):
            config[curr_config]["completed"] = False
            with open('results/config.json', 'w') as f:
                json.dump(config, f, indent=4)

            rejected_configs.append(curr_config)
            continue

        if END_TIME - time.time() < 0.3*60*60:
            config[curr_config]["completed"] = False
            with open('results/config.json', 'w') as f:
                json.dump(config, f, indent=4)
            break


        # training script
        model.args.learning_rate = learning_rate
        model.args.num_train_epochs = epoch
        model.args.per_device_train_batch_size = batch_size
        
        print(f"running: {curr_config}")
        start = time.time()
        model.train()
        results = model.evaluate()
        os.system(f"rm {model.args.output_dir}/* -rf")
        runtime = float(time.time() - start)/3600
        print(f"{curr_config} final results")
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

        with open('results/config.json', 'w') as f:
            json.dump(config, f, indent=4)
    print("Script done")
if __name__ == "__main__":
    main()