import copy
import json
import os
from os import path


def multiset_sub(a, b):
    """Compute multiset `a - b` and return the result. This is immutable and modifies neither a nor b."""
    a = copy.deepcopy(a)
    for elt in b:
        if elt in a:
            a.remove(elt)

    return a


def multiset_intersection(a, b):
    """Compute multiset intersection `a ∩ b` and return the result. This is immutable and modifies neither a nor b."""
    b = copy.deepcopy(b)

    intersection = []

    for elt in a:
        if elt in b:
            b.remove(elt)
            intersection.append(elt)

    return intersection


def eval_model_aligned_multiset(predicted_tags, target_tags):
    results_per_tag = dict()
    default_entry = {"false_pos": 0, "true_pos": 0, "false_neg": 0}

    for prediction, target in zip(predicted_tags, target_tags):
        # According to Seker & Tsafarty, 2020:
        # - |true_pos| of token = |pred ∩ gold|
        # - |false_pos| of token = |pred - gold|
        # - |false_neg| of token = |gold - pred|

        true_pos = multiset_intersection(prediction, target)
        false_pos = multiset_sub(prediction, target)
        false_neg = multiset_sub(target, prediction)

        for tag in true_pos:
            results_per_tag.setdefault(tag, copy.deepcopy(default_entry))
            results_per_tag[tag]["true_pos"] += 1

        for tag in false_pos:
            results_per_tag.setdefault(tag, copy.deepcopy(default_entry))
            results_per_tag[tag]["false_pos"] += 1

        for tag in false_neg:
            results_per_tag.setdefault(tag, copy.deepcopy(default_entry))
            results_per_tag[tag]["false_neg"] += 1

    f1_per_tag = dict()
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0
    for tag, results in results_per_tag.items():
        true_pos = results["true_pos"]
        false_pos = results["false_pos"]
        false_neg = results["false_neg"]

        # Add to global stats for micro averaging
        total_true_pos += true_pos
        total_false_pos += false_pos
        total_false_neg += false_neg

        # Calculate per-tag F1
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_per_tag[tag] = f1

    macro_f1 = sum(f1_per_tag.values()) / len(f1_per_tag)
    micro_precision = total_true_pos / (total_true_pos + total_false_pos)
    micro_recall = total_true_pos / (total_true_pos + total_false_neg)
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    return micro_f1, macro_f1


def eval_model_aligned_set(predicted_tags, target_tags):
    results_per_tag = dict()
    default_entry = {"tagger_produced": 0, "gold_standard_produced": 0, "tagger_correct": 0}

    for prediction, target in zip(predicted_tags, target_tags):
        for i in range(len(prediction)):
            pred_tag = prediction[i]
            target_tag = target[i] if i < len(target) else None

            results_per_tag.setdefault(pred_tag, copy.deepcopy(default_entry))
            results_per_tag.setdefault(target_tag, copy.deepcopy(default_entry))

            results_per_tag[pred_tag]["tagger_produced"] += 1

            if target_tag is not None:
                results_per_tag[target_tag]["gold_standard_produced"] += 1

            if pred_tag == target_tag:
                results_per_tag[pred_tag]["tagger_correct"] += 1

        for i in range(len(prediction), len(target)):
            target_tag = target[i]
            results_per_tag.setdefault(target_tag, copy.deepcopy(default_entry))
            results_per_tag[target_tag]["gold_standard_produced"] += 1

    f1_per_tag = dict()
    total_gold_standard_produced = 0
    total_tagger_produced = 0
    total_correct = 0
    for tag, results in results_per_tag.items():
        gold_standard_produced = results["gold_standard_produced"]
        tagger_produced = results["tagger_produced"]
        correct = results["tagger_correct"]

        # Add to global stats for micro averaging
        total_gold_standard_produced += gold_standard_produced
        total_tagger_produced += tagger_produced
        total_correct += correct

        # Calculate per-tag F1
        precision = correct / tagger_produced if tagger_produced > 0 else 0
        recall = correct / gold_standard_produced if gold_standard_produced > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_per_tag[tag] = f1

    macro_f1 = sum(f1_per_tag.values()) / len(f1_per_tag)
    micro_precision = total_correct / total_tagger_produced
    micro_recall = total_correct / total_gold_standard_produced
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    return micro_f1, macro_f1


class ModelResults:
    def __init__(self, predicted_output_filename, seed, language, model, context, tokenization, segmentation,
                 is_from_scratch):
        self.predicted_output_filename = predicted_output_filename
        self.seed = seed
        self.language = language
        self.model = model
        self.context = context
        self.tokenization = tokenization
        self.segmentation = segmentation
        self.is_from_scratch = is_from_scratch
        self.train_type = "from_scratch" if is_from_scratch else "plm"

    @staticmethod
    def parse_from_scratch(saved_filename: str, dir_path: str):
        predicted_output_filename = path.join(dir_path, f"results-new-{saved_filename.removesuffix('.pt')}.txt")
        seed = int(saved_filename.split("seed-")[1][0])
        language = saved_filename.split("-seed")[0][-2:]

        # Easier to split by - now
        name = saved_filename.replace("character-summing", "character_summing")
        name = name.replace("bilstm-crf", "bilstm_crf")
        model, context, tokenization = name.split("-")[:3]
        model = model.replace("bilstm_crf", "CRF").replace("bilstm", "BiLSTM")
        tokenization = tokenization.replace("character_summing", "character-sum")

        if "lower-surface" in name:
            tokenization = tokenization + "-lower"

        if "surface" in name:
            segmentation = "surface"
        else:
            segmentation = "canonical"

        return ModelResults(predicted_output_filename, seed, language, model, context, tokenization, segmentation, True)

    @staticmethod
    def parse_plm(saved_filename: str, dir_path: str):
        MODEL_NAMES = {"a": "Afro-XLMR", "n": "Nguni-XLMR", "x": "XLM-R-large"}

        predicted_output_filename = path.join(dir_path, saved_filename)
        split = saved_filename.removesuffix(".tsv").split("_")
        seed = int(split[2])
        language = split[0]
        model = MODEL_NAMES[split[1]]

        context = "words"
        tokenization = ""

        if "canon" in dir_path:
            segmentation = "canonical"
        else:
            segmentation = "surface"

        return ModelResults(predicted_output_filename, seed, language, model, context, tokenization, segmentation,
                            False)

    def results(self):
        with open(self.predicted_output_filename) as f:
            gold_pred, gold_targ, predseg_pred = [], [], []

            for i, line in enumerate(f):
                line = line.strip()

                if not line or line.startswith("morphemes\ttarget\tprediction"):
                    continue

                line = line.replace("<?word_sep?>", "<?word-sep?>")
                split = line.split("\t")

                # PLMs sometimes failed to produce anything - unfortunate :(
                if len(split) == 4 and split[2] == "":
                    # print(self.predicted_output_filename.split("/")[-1], "Line", i + 1, line)
                    _, target, prediction, _ = split
                    predseg_prediction = ""
                elif len(split) == 4 and all(s != "" for s in split):
                    # print(self.predicted_output_filename.split("/")[-1], "Line", i + 1, line)
                    _, target, prediction, _ = split
                    predseg_prediction = ""
                else:
                    _, target, prediction, _, predseg_prediction = split

                gold_targ.append(target.split("_"))
                gold_pred.append(prediction.split("_"))
                predseg_pred.append(predseg_prediction.split("_"))

        results = {
            "aligned_set": {"predicted_segmentation": {}, "gold_segmentation": {}},
            "aligned_multiset": {"predicted_segmentation": {}, "gold_segmentation": {}}
        }

        gold_micro, gold_macro = eval_model_aligned_set(gold_pred, gold_targ)

        pred_micro, pred_macro = eval_model_aligned_set(predseg_pred, gold_targ)
        results["aligned_set"]["gold_segmentation"]["micro"] = gold_micro
        results["aligned_set"]["gold_segmentation"]["macro"] = gold_macro
        results["aligned_set"]["predicted_segmentation"]["micro"] = pred_micro
        results["aligned_set"]["predicted_segmentation"]["macro"] = pred_macro

        gold_micro, gold_macro = eval_model_aligned_multiset(gold_pred, gold_targ)
        pred_micro, pred_macro = eval_model_aligned_multiset(predseg_pred, gold_targ)
        results["aligned_multiset"]["gold_segmentation"]["micro"] = gold_micro
        results["aligned_multiset"]["gold_segmentation"]["macro"] = gold_macro
        results["aligned_multiset"]["predicted_segmentation"]["micro"] = pred_micro
        results["aligned_multiset"]["predicted_segmentation"]["macro"] = pred_macro

        return results


def process_model_results(folder, is_from_scratch, plm_type_key=None):
    all_seeds_by_lang = dict()
    example_model = None
    for file in os.listdir(folder):
        if not path.isfile(path.join(folder, file)) or (is_from_scratch and not file.endswith(".pt")):
            continue

        if plm_type_key is not None and file.split("_")[1] != plm_type_key:
            continue

        if is_from_scratch:
            model = ModelResults.parse_from_scratch(file, folder)
        else:
            model = ModelResults.parse_plm(file, folder)

        all_seeds_by_lang.setdefault(model.language, []).append(model.results())
        example_model = model

    results_by_lang = dict()
    for lang, lang_results in all_seeds_by_lang.items():
        results_by_lang[lang] = dict()
        for eval_type in ("aligned_set", "aligned_multiset"):
            results_by_lang[lang][eval_type] = dict()
            for seg_type in ("predicted_segmentation", "gold_segmentation"):
                results_by_lang[lang][eval_type][seg_type] = mean_average_results(lang_results, eval_type, seg_type)

    return (example_model, results_by_lang)


def mean_average_results(seed_results, eval_type, seg_type):
    # TODO Filter to seeds which initialised properly
    seed_results = [
        result[eval_type][seg_type]
        for result in seed_results
        if result[eval_type][seg_type]["macro"] > 0.1
    ]

    micro_total = sum(result["micro"] for result in seed_results)
    macro_total = sum(result["macro"] for result in seed_results)
    return {"micro": micro_total / len(seed_results), "macro": macro_total / len(seed_results)}


def get_best_in_lang(results, measure, lang_key, eval_type, seg_type):
    best_model = None
    best_score = 0.0
    for context_key, context_res in results.items():
        for model_name, model_res in context_res.items():
            model_score = model_res[lang_key][eval_type][seg_type][measure]
            if model_score > best_score:
                best_model = model_name, context_key
                best_score = model_score
    return best_model


def recalculate_results():
    all_results = dict()

    # Load canonical PLM results
    for plm_type_key in ("a", "n", "x"):
        for seg_type in ("canon", "surface"):
            model, results = process_model_results(f"out_models/plms/{seg_type}", False, plm_type_key)
            print(model.model)

            all_results.setdefault(model.segmentation, dict())
            all_results[model.segmentation].setdefault(model.train_type, dict())
            all_results[model.segmentation][model.train_type].setdefault(model.context, dict())
            all_results[model.segmentation][model.train_type][model.context][model.model] = results

    # Load from-scratch results
    for folder in os.listdir("out_models/new/"):
        if os.path.isdir(os.path.join("out_models/new/", folder)):
            print(folder)
            model, results = process_model_results(f"out_models/new/{folder}", True)
            all_results.setdefault(model.segmentation, dict())
            all_results[model.segmentation].setdefault(model.train_type, dict())
            all_results[model.segmentation][model.train_type].setdefault(model.context, dict())
            name = f"{model.model}, {model.tokenization}"
            all_results[model.segmentation][model.train_type][model.context][name] = results

    with open("results_cache.json", "w") as f:
        f.write(json.dumps(all_results, indent=2))


def main():
    # recalculate_results()

    with open("results_cache.json", "r") as f:
        all_results = json.loads(f.read())

    for seg_type in ("gold_segmentation", "predicted_segmentation"):
        print("=" * 50)
        print(f" {seg_type} ")
        print("=" * 50)

        for surface_or_canon in ("canonical", "surface"):
            if seg_type == "predicted_segmentation":
                if surface_or_canon == "surface":
                    print(
                        "\\midrule\n"
                        "\\multicolumn{9}{c}{\\textbf{Surface segmentations on data from \\citet{morph-segment-jan}}}"
                        " \\\\ \n"
                    )
                else:
                    print(
                        "\\toprule\n"
                        "\\textbf{Model} & \\multicolumn{2}{c}{\\textbf{IsiZulu}} & \\multicolumn{2}{c}{\\textbf{IsiNdebele}} & "
                        "\\multicolumn{2}{c}{\\textbf{IsiXhosa}} & \\multicolumn{2}{c}{\\textbf{Siswati}} \\\\ \n"

                        "\\midrule\n"

                        "& Mac $F_1$ & Mic $F_1$ & Mac $F_1$ & Mic $F_1$ & Mac $F_1$ & Mic $F_1$ & Mac $F_1$ &"
                        " Mic $F_1$ \\\\"
                    )

                    print(
                        "\\midrule\n"
                        "\\multicolumn{9}{c}{\\textbf{Canonical segmentations on data from \\citet{morph-segment-jan}}}"
                        " \\\\ \n"
                    )
            else:
                print("-" * 20, surface_or_canon, "-" * 20)

                print(
                    "\\toprule\n"
                    "\\textbf{Model} & \\multicolumn{2}{c}{\\textbf{IsiZulu}} & \\multicolumn{2}{c}{\\textbf{IsiNdebele}} & "
                    "\\multicolumn{2}{c}{\\textbf{IsiXhosa}} & \\multicolumn{2}{c}{\\textbf{Siswati}} \\\\ \n"

                    "\\midrule\n"

                    "& Mac $F_1$ & Mic $F_1$ & Mac $F_1$ & Mic $F_1$ & Mac $F_1$ & Mic $F_1$ & Mac $F_1$ &"
                    " Mic $F_1$ \\\\"
                )

            results = all_results[surface_or_canon]

            langs = (("IsiZulu", "ZU"), ("IsiNdebele", "NR"), ("IsiXhosa", "XH"), ("SiSwati", "SS"))
            train_types = (("Models trained from-scratch", "from_scratch"), ("Pre-trained language models", "plm"))

            for train_type, train_type_key in train_types:
                print(
                    "\\midrule\n"
                    f"\\textbf{{{train_type}}} & & & & & & & & \\\\\n"
                    "\\midrule"
                )

                for context_name, context_key in (("Word-level", "words"), ("Sentence-level", "sentences")):
                    if surface_or_canon == "surface" and context_key == "words" and train_type_key == "from_scratch":
                        continue

                    if context_key == "sentences" and train_type_key == "plm":
                        continue

                    print(f"\\textbf{{{context_name}}} & & & & & & & & \\\\")

                    models_sorted = sorted(results[train_type_key][context_key].items(), key=lambda kv: kv[0])
                    for model_name, model_res in models_sorted:
                        print(model_name, end=" & ")

                        for lang_name, lang_key in langs:
                            eval_ty = "aligned_multiset"
                            best_macro = get_best_in_lang(results[train_type_key], "macro", lang_key, eval_ty, seg_type)
                            best_micro = get_best_in_lang(results[train_type_key], "micro", lang_key, eval_ty, seg_type)

                            lang_res = model_res[lang_key]["aligned_multiset"][seg_type]
                            macro = lang_res["macro"]
                            micro = lang_res["micro"]

                            macro_fmt = f"{macro * 100:.1f}"
                            micro_fmt = f"{micro * 100:.1f}"
                            if best_macro == (model_name, context_key):
                                macro_fmt = f"\\textbf{{{macro_fmt}}}"

                            if best_micro == (model_name, context_key):
                                micro_fmt = f"\\textbf{{{micro_fmt}}}"

                            print(f"{macro_fmt} & {micro_fmt}", end="")
                            if lang_key != langs[-1][1]:
                                print(" & ", end="")

                        print("\\\\")

            if not (seg_type == "predicted_segmentation" and surface_or_canon == "canonical"):
                print("\\bottomrule")


if __name__ == "__main__":
    main()
