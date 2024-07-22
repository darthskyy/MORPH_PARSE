import json
import os
import pprint
import time
import random


def create_config_file(filename: str = "results/config.json"):
    if not os.path.exists(filename):
        configs = {}
        # these are the main ones for my grid search, can be altered to suit other needs
        for model in ["Davlan/afro-xlmr-large-76L", "xlm-roberta-large"]:
            for language in ["NR", "SS", "XH", "ZU"]:
                for learning_rate in [1e-5, 3e-5, 5e-5]:
                    for batch_size in [16, 32]:
                        for epoch in [5, 10, 15]:
                            config = {
                                "id": f"{language}_{str(learning_rate)[0]}_{batch_size}_{epoch}_{model.split('/')[-1][0]}",
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
                            }
                            configs[config["id"]] = config
        with open(filename, "w") as f:
            json.dump(configs, f, indent=4)


def import_csv(file_path, config_file):
    if not os.path.exists(config_file):
        create_config_file(config_file)
    lines = []
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]

    configs = {}
    with open(config_file, "r") as f:
        configs = json.load(f)

    for line in lines:
        parts = line.split(",")
        language = parts[0]
        model = parts[1]
        lr = float(parts[2])
        epochs = int(parts[3])
        batch_size = int(parts[4])
        loss = float(parts[5])
        f1 = float(parts[6])
        precision = float(parts[7])
        recall = float(parts[8])
        runtime = float(parts[9])
        id_ = f"{language}_{str(lr)[0]}_{batch_size}_{epochs}_{model.split('/')[-1][0]}"
        if id_ in configs:
            if f1 < configs[id_]["macro_f1"]:
                continue
            configs[id_]["loss"] = loss
            configs[id_]["macro_f1"] = f1
            configs[id_]["macro_precision"] = precision
            configs[id_]["macro_recall"] = recall
            configs[id_]["runtime"] = runtime
            configs[id_]["completed"] = True
            configs[id_]["timestamp"] = time.time()

    with open(config_file, "w") as f:
        json.dump(configs, f, indent=4)


if __name__ == "__main__":
    with open("results/config.json", "r") as f:
        configs = json.load(f)
    for key in list(configs.keys()):
        if "_8_" in key:
            del configs[key]
    
    for model in ["francois-meyer/nguni-xlmr-large"]:
        for language in ["NR", "SS", "XH", "ZU"]:
            for learning_rate in [1e-5, 3e-5, 5e-5]:
                for batch_size in [16, 32]:
                    for epoch in [5, 10, 15]:
                        config = {
                            "id": f"{language}_{str(learning_rate)[0]}_{batch_size}_{epoch}_{model.split('/')[-1][0]}",
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
                        }
                        configs[config["id"]] = config
    
    with open("results/config.json", "w") as f:
        json.dump(configs, f, indent=4)
