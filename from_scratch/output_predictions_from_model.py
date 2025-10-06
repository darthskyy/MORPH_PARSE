import os

from demo import *
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def eval_model_on_set(model, testset_suffix):
    """Evaluate a given model on the given testset"""

    model.eval()


    with torch.no_grad():

        path = f"data/TEST/{model.lang}_TEST{testset_suffix}.tsv"
        raw = extract_morphemes_and_tags_from_file_2022(path, use_surface=model.is_surface, is_demo=True)
        test_set = list(split_sentences_raw(raw))

        print(eval_model_aligned_multiset(model, test_set))

        all_morphemes, gold, pred = [], [], []
        for i, (morphemes, gold_tags) in enumerate(test_set):
            if not morphemes:
                continue

            # Break up per word to provide it in a list-of-lists format to the model
            # The model still gets full sentence level context if it needs it
            gold_tags_per_word = list(split_words(gold_tags))
            morphemes_per_word = list(split_words(morphemes))

            all_morphemes.extend(morphemes_per_word)
            gold.extend(gold_tags_per_word)
            pred_tags = model.forward(morphemes_per_word)
            pred.extend(pred_tags)

        return all_morphemes, gold, pred


def output_results(model_dir: str, model_filename: str):
    model: EncapsulatedModel = torch.load(os.path.join(model_dir, model_filename), map_location=device)
    model.eval()

    results_out_path = os.path.join(model_dir, "results-new-" + model_filename.replace(".pt", ".txt"))

    with torch.no_grad(), open(results_out_path, "w") as f:
        f.write("morphemes\ttarget\tprediction\tmorphemes_predseg\tprediction_predseg\n")
        suffix_gold, suffix_pred = ("SET_GOLD_SURFACE", "_SURFACE") if model.is_surface else ("", "_CANONICAL_PRED")

        morphemes_gold, target_gold, pred_gold = eval_model_on_set(model, suffix_gold)
        morphemes_predseg, _, pred_predseg = eval_model_on_set(model, suffix_pred)

        for columns in zip(morphemes_gold, target_gold, pred_gold, morphemes_predseg, pred_predseg):
            f.write("\t".join(["_".join(col) for col in columns]))
            f.write("\n")


def process_model_results(folder):
    for file in os.listdir(folder):
        if not os.path.isfile(os.path.join(folder, file)) or not file.endswith(".pt"):
            continue

        if not file.startswith("bilstm-words-character-summing"):
            continue

        print(file)
        output_results(folder, file)


def main():
    for folder in os.listdir("out_models/new/"):
        if os.path.isdir(os.path.join("out_models/new/", folder)):
            process_model_results(f"out_models/new/{folder}")


if __name__ == "__main__":
    main()
