import copy
import re

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from from_scratch import encapsulated_model, lstm, bilstm_crf, dataset, common
from .dataset import (split_sentences_raw, extract_morphemes_and_tags_from_file_2022, WORD_SEP_TEXT,
                     SEQ_PAD_TEXT, identity)
from .encapsulated_model import EncapsulatedModel
from .aligned_f1 import align_seqs
import sys

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tag_pattern = re.compile(r'\[[a-zA-Z-_0-9|]*?]-?')


def segment_query(q):
    """Segment a query into its morphemes. Users can either input the entire (tagged) sequence from the dataset,
    in which case the tags will be stripped, or they can just separate morphemes by hyphenation."""

    if tag_pattern.search(q):
        return [[morpheme for morpheme in tag_pattern.split(word) if morpheme != ""] for word in q.split(" ")]
    else:
        return [word.split("-") for word in q.split(" ")]


def split_words(seq):
    """Split a sentence into its words"""

    word = []
    for i in seq:
        if i != WORD_SEP_TEXT:
            word.append(i)
        else:
            yield word
            word = []
    if word:
        yield word


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


def eval_model_aligned_multiset(model, test_set):
    results_per_tag = dict()
    default_entry = {"false_pos": 0, "true_pos": 0, "false_neg": 0}

    for morphemes, gold_tags in test_set:
        if not morphemes:
            continue

        # Break up per word to provide it in a list-of-lists format to the model
        # The model still gets full sentence level context if it needs it
        gold_tags_per_word = list(split_words(gold_tags))
        morphemes_per_word = list(split_words(morphemes))

        for prediction, target in zip(model.forward(morphemes_per_word), gold_tags_per_word):
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


def eval_model_aligned_set(model, test_set):
    results_per_tag = dict()
    default_entry = {"tagger_produced": 0, "gold_standard_produced": 0, "tagger_correct": 0}

    for morphemes, gold_tags in test_set:
        if not morphemes:
            continue

        # Break up per word to provide it in a list-of-lists format to the model
        # The model still gets full sentence level context if it needs it
        gold_tags_per_word = list(split_words(gold_tags))
        morphemes_per_word = list(split_words(morphemes))

        for prediction, target in zip(model.forward(morphemes_per_word), gold_tags_per_word):
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


def eval_model(model, test_set, map_tag=identity):
    """Evaluate a given model on the testset, returning its performance"""

    with open("out.txt", "w") as f:
        f.write("morphemes\ttarget\tprediction\n")

        gold, pred = [], []
        for morphemes, gold_tags in test_set:
            if not morphemes:
                continue

            # Break up per word to provide it in a list-of-lists format to the model
            # The model still gets full sentence level context if it needs it
            gold_tags_per_word = list(split_words(gold_tags))
            morphemes_per_word = list(split_words(morphemes))

            for morphemes_word, pred_word, gold_word in zip(morphemes_per_word, model.forward(morphemes_per_word), gold_tags_per_word):
                print("_".join(morphemes_word), "_".join(gold_word), "_".join(pred_word), sep="\t", file=f)
                pred_word, gold_word = align_seqs(pred_word, gold_word)

                for (expected_tag, predicted_tag) in zip(gold_word, pred_word):
                    if expected_tag == WORD_SEP_TEXT or expected_tag == SEQ_PAD_TEXT:
                        continue

                    pred.append(map_tag(predicted_tag))
                    gold.append(map_tag(expected_tag))

        micro = f1_score(gold, pred, average="micro")
        macro = f1_score(gold, pred, average="macro")
        report = classification_report(gold, pred, zero_division=0.0)
        return micro, macro, report


def _with_sys_modules(func, **modules):
    old_modules = dict()
    for module_name, module in modules.items():
        if module_name in sys.modules:
            old_modules[module_name] = sys.modules[module_name]
        sys.modules[module_name] = module

    ret = func()

    for module_name, module in modules.items():
        if module_name in old_modules:
            sys.modules[module_name] = old_modules[module_name]

    return ret


def load_model(path):
    model: EncapsulatedModel = _with_sys_modules(
        lambda: torch.load(path, map_location=device, weights_only=False),
        encapsulated_model=encapsulated_model,
        lstm=lstm,
        bilstm_crf=bilstm_crf,
        dataset=dataset,
        common=common,
    )
    model.eval()
    return model


def annotate_sentence(model, words):
    with torch.no_grad():
        annotated_words = [list(zip(word, tags)) for word, tags in zip(words, model.forward(words))]
        return ["-".join(f"{morpheme}[{tag}]" for morpheme, tag in word) for word in annotated_words]


def predict_tags_for_word(model, morphemes):
    with torch.no_grad():
        # Return the first sentence's first word (we have one word per sentence, which there is also one of, here)
        return model.forward([[morphemes]])[0][0]


def predict_tags_for_words_batched(model, words):
    """Predict the tags for a list of _unrelated_ words (i.e. batch, not together in a sentence)"""
    with torch.no_grad():
        # Wrap each word in a list so it is, itself, treated as a sentence
        tags = model.forward([[word] for word in words])

        # Return the first word of each sentence (we have one word per sentence here)
        return [sentence[0] for sentence in tags]


"""
seed 0
XH Micro F1: 0.9526032080 (0.9526). Macro F1: 0.7409512152 (0.7410)
Aligned set micro: 0.9561200923787528 macro: 0.7441283833000646
Aligned multiset micro: 0.9565160013196965

seed 1
XH Micro F1: 0.9547725480 (0.9548). Macro F1: 0.7524040931 (0.7524)
Aligned set micro: 0.9582975915539426 macro: 0.7556810496988166
Aligned multiset micro: 0.9586935004948861

seed 2
XH Micro F1: 0.9548382856 (0.9548). Macro F1: 0.7571155235 (0.7571)
Aligned set micro: 0.9583635763774331 macro: 0.7603705099568765
Aligned multiset micro: 0.9586275156713956

seed 3
XH Micro F1: 0.9545753353 (0.9546). Macro F1: 0.7489831197 (0.7490)
Aligned set micro: 0.9580996370834708 macro: 0.7522593772445954
Aligned multiset micro: 0.9582975915539426

seed 4
XH Micro F1: 0.9518143571 (0.9518). Macro F1: 0.7451841447 (0.7452)
Aligned set micro: 0.9553282744968656 macro: 0.7484124590309923
Aligned multiset micro: 0.9556581986143187

"""

def demo():
    while True:
        # path = input("Model > ")
        path = "new/bilstm-sentences-character-summing/bilstm-sentences-character-summing-NR-seed-1-final-epoch-20.pt"
        model: EncapsulatedModel = torch.load("out_models/" + path, map_location=device)
        model.eval()

        with torch.no_grad():
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print(f"Model has {params / 1_000_000:.2f}M parameters")

            suffix = "_SURFACE" if model.is_surface else ""
            # suffix = "_CANONICAL_PRED"  # <-- uncomment this and comment the above if you want to use predicted canonical segmentations
            path = f"data/TEST/{model.lang}_TEST{suffix}.tsv"
            sentences = list(split_sentences_raw(extract_morphemes_and_tags_from_file_2022(path, use_surface=model.is_surface, is_demo=True)))

            micro, macro, report = eval_model(model, sentences)
            print(report)
            print(f"{model.lang} Micro F1: {micro:.10f} ({micro:.4f}). Macro F1: {macro:.10f} ({macro:.4f})")

            micro, macro = eval_model_aligned_set(model, sentences)
            print("Aligned set micro:", micro, "macro:", macro)

            micro, macro = eval_model_aligned_multiset(model, sentences)
            print("Aligned multiset micro:", micro, "macro:", macro)

            while True:
                query = input("Morphological segmentation (separated by -) or Q to quit > ")
                if query.lower().strip() == "q":
                    break
                words = segment_query(query)

                annotated_words = [list(zip(word, tags)) for word, tags in zip(words, model.forward(words))]
                print(" ".join("-".join(f"{morpheme}[{tag}]" for morpheme, tag in word) for word in annotated_words))
        break


if __name__ == "__main__":
    demo()
