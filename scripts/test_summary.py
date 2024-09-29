"""
File for summarising the final test results of the project run in seeds of five
"""

import argparse
import json
import itertools
import sys

def main():
    parser = argparse.ArgumentParser("Parser for test evaluations summariser")
    parser.add_argument("--results_file", required=False, default="results/final.json", type=str, help="file containing the final test results results.")
    args = parser.parse_args()

    print(f"results from {args.results_file}")
    results = json.load(open(args.results_file, "r"))

    LANGUAGES = ["NR", "SS", "XH", "ZU"]
    MODELS = ["xlm-roberta-large", "Davlan/afro-xlmr-large-76L", "francois-meyer/nguni-xlmr-large"]

    pairs = [(model["language"], model["model"]) for _, model in results.items()]
    pairs = list(itertools.product(LANGUAGES, MODELS))

    micros = {
        pair: [results[item]["micro_f1"] for item in results if results[item]["language"] == pair[0] and results[item]["model"] == pair[1] and results[item]["macro_f1"]>0.2]
        for pair in pairs
    }

    macros = {
        pair: [results[item]["macro_f1"] for item in results if results[item]["language"] == pair[0] and results[item]["model"] == pair[1] and results[item]["macro_f1"]>0.2]
        for pair in pairs
    }

    print(f"{'language':15}{'micro mean':<25}{'micro best':<25}{'macro mean':<25}{'macro best':<25}{'seed':7}{'model'}")

    old_language = pairs[0][0]
    for pair in pairs:
        if pair[0] != old_language:
            old_language = pair[0]
            print()
        output = f"{pair[0]:15}"
        output += f"{sum(micros[pair])/len(micros[pair]):<25.4f}" if len(micros[pair]) > 0 else f"{'null':25}"
        output += f"{max(micros[pair]):<25.4f}" if len(micros[pair]) > 0 else f"{'null':25}"
        output += f"{sum(macros[pair])/len(macros[pair]):<25.4f}" if len(micros[pair]) > 0 else f"{'null':25}"
        output += f"{max(macros[pair]):<25.4f}" if len(micros[pair]) > 0 else f"{'null':25}"
        output += f"{str(len(macros[pair])) + '/5':<7}"
        output += f"{pair[1]}"

        print(output)


    print("#"*150)
    print("Completion")
    seed_1 = [results[item] for item in results if results[item]["seed"] in ["1", 1]]

    seeds = [
        [results[item] for item in results if results[item]["seed"] in [f"{seed}", seed]]
        for seed in range(1,6)
    ]

    languages = [
        [results[item] for item in results if results[item]["language"] in [language]]
        for language in LANGUAGES
    ]

    models = [
        [results[item] for item in results if results[item]["model"] in [model]]
        for model in MODELS
    ]

    for i in range(5):
        output = f"Seed {seeds[i][0]['seed']}: "
        output += str(len([1 for item in seeds[i] if item["completed"] == True])) +  "/" + str(len(seeds[i]))
        if i < len(LANGUAGES):
            output += f"\t\tLanguage {languages[i][0]['language']}: "
            output += str(len([1 for item in languages[i] if item["completed"] == True])) +  "/" + str(len(languages[i]))

        if i < len(MODELS):
            output += f"\t\tModel {models[i][0]['model']:31}: "
            output += str(len([1 for item in models[i] if item["completed"] == True])) +  "/" + str(len(models[i]))
        print(output)

    print("#"*150)
    print("Running")

    running_models = [item for item in results if results[item]["completed"] in ["running"]]

    for model in running_models:
        print(model)

if __name__ == "__main__":
    main()
