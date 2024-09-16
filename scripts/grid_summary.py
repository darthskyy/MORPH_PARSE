from datetime import datetime
# checking for completion in the tests
# pylint: disable=C0301
import json
import argparse
import sys

def get_percentage(search: str, search_space: list):
    primary = [item for item in search_space if search in item["id"]]

    return round(len([item for item in primary if item["completed"]])/len(primary) *100)


def print_grid_search_summary(args):
    config = json.load(open(args.results_file))
    # key for sorting by macro_f1
    date_format = "%d %B, %Y %H:%M"

    ## Dealing with model figures
    models = ["xlm-roberta-large", "Davlan/afro-xlmr-large-76L", "francois-meyer/nguni-xlmr-large"]
    languages = ["NR", "SS", "XH", "ZU"]
    code_names = {
        "xlm-roberta-large": "XLMR",
        "Davlan/afro-xlmr-large-76L": "AfroXLMR",
        "francois-meyer/nguni-xlmr-large": "NguniXLMR",
    }
    language_names = {
        "NR": "isiNdebele",
        "SS": "siSwati",
        "XH": "isiXhosa",
        "ZU": "isiZulu",
    }
    trials = {
        model:sorted([config[item] for item in config if config[item]["model"]==model], key=lambda x: x["macro_f1"], reverse=True)
        for model in models
    }

    # stored as model: (completed, total)
    trials_figures = {
        model: (len([x for x in trials[model] if x["completed"]]), len(trials[model]))
        for model in models
    }

    print("models")
    for model in models:
        out_string = f"{code_names[model]:<10}: {trials_figures[model][0]}/{trials_figures[model][1]}\t"
        out_string += f"Best Trial: {trials[model][0]['id']:<12} with {trials[model][0]['macro_f1']*100:.2f}% on {datetime.fromtimestamp(trials[model][0]['timestamp']).strftime(date_format)}\t"
        out_string += f"({get_percentage('NR', trials[model])}%NR, {get_percentage('SS', trials[model])}%SS, {get_percentage('XH', trials[model])}%XH, {get_percentage('ZU', trials[model])}%ZU)"
        print(out_string)


    ## Dealing with Languages
    language_trials = {
        language: sorted([config[item] for item in config if config[item]["language"]==language], key=lambda x: x["macro_f1"], reverse=True)
        for language in languages
    }

    language_figures = {
        language: (len([x for x in language_trials[language] if x["completed"]]), len(language_trials[language]))
        for language in languages
    }

    print()
    print("languages")
    for language in languages:
        out_string = f"{language_names[language]:<10}: {language_figures[language][0]}/{language_figures[language][1]}\t"
        out_string += f"Best Trial: {language_trials[language][0]['id']:<12} with {language_trials[language][0]['macro_f1']*100:.2f}% "
        out_string += f"on {datetime.fromtimestamp(language_trials[language][0]['timestamp']).strftime(date_format)}"
        print(out_string)

    print()
    print("most recent runs")
    recent_trials = sorted([config[item] for item in config if config[item]["micro_f1"]>0], key=lambda x: x["timestamp"], reverse=True)[:5]
    for trial in recent_trials:
        print(f'{trial["id"]:<15}{datetime.fromtimestamp(trial["timestamp"]).strftime(date_format)}\t{trial["runtime"]:<.2f} hours')

    running_trials = [config[item] for item in config if config[item]["completed"]=="running"]
    if running_trials:
        print()
        print("running trials")
        for trial in running_trials:
            print(f'{trial["id"]:<15}')

    if not args.save_csv:
        sys.exit()

    lines = [] 
    for run in config.keys():
        if "_8_" in run: continue
        lines.append(",".join([str(item) for item in list(config[run].values())]))

    headings = [",".join(config[run].keys())]

    lines = headings + lines
    lines = "\n".join(lines)
    with open(args.save_csv, "w", encoding="utf-8") as f:
        f.write(lines)


def main():
    parser = argparse.ArgumentParser("Parser for grid parse summariser")
    parser.add_argument("--save_csv", required=False, type=str, help="file to save the results in csv format.")
    parser.add_argument("--results_file", required=False, default="results/config.json", type=str, help="file containing the grid search results.")
    args = parser.parse_args()

    print_grid_search_summary(args=args)


if __name__=="__main__":
    main()
