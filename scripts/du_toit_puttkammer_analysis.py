import csv
import math

from sklearn.metrics import f1_score, classification_report

from aligned_f1 import align_seqs
from data_prep import split_tags
from from_scratch.dataset import tags_only_no_classes

# for lang in ("ZU", "NR", "XH", "SS"):
for lang in (("NR", "XH", "SS", "ZU")):
    with open(f"data/DuToitPuttkammer/MorphAnalysis.{lang}.{lang}_TEST_OUT.txt") as f:
        pred_lines = [line.split("\t") for line in f.read().splitlines(keepends=False) if line]

    with open(f"data/TEST/{lang}_TEST.tsv") as f:
        gold_lines = [line.split("\t") for line in f.read().splitlines(keepends=False) if line]

    # with open(f"data/DuToitPuttkammer/MorphAnalysis.{lang}.{lang}_TRAIN.txt") as f:
    #     pred_lines = [line.split("\t") for line in f.read().splitlines(keepends=False) if line]
    #
    # with open(f"data/TRAIN/{lang}_TRAIN.tsv") as f:
    #     gold_lines = [line.split("\t") for line in f.read().splitlines(keepends=False) if line]


    all_gold_seg, all_pred_seg = [], []
    all_gold_tag, all_pred_tag = [], []
    all_gold, all_pred = [], []

    seen_raw = set()

    correct = 0
    total = 0

    for (i, (pred_line, gold_line)) in enumerate(zip(pred_lines, gold_lines)):
        pred_raw, pred_analysis = pred_line
        gold_raw, _, gold_seg, gold_tags = gold_line
        gold_seg = gold_seg.split("_")
        gold_tags = [tags_only_no_classes(tag) for tag in gold_tags.split("_")]
        assert pred_raw.lower() == gold_raw.lower(), f"{lang} {i}: {pred_raw}, {gold_raw}"
        pred_seg, pred_tags = split_tags(pred_analysis)

        if gold_tags[0] in ["Punc", "Num"]:
            continue

        # if gold_raw.lower() in seen_raw:
        #     continue
        # seen_raw.add(gold_raw.lower())

        pred_comb = [f"{seg}[{tag}]" for seg, tag in zip(pred_seg, pred_tags)]
        gold_comb = [f"{seg}[{tag}]" for seg, tag in zip(gold_seg, gold_tags)]

        for p in pred_comb:
            total += 1
            if p in gold_comb:
                correct += 1
        total += abs(len(gold_comb) - len(pred_comb))

        pred_seg, gold_seg = align_seqs(pred_seg, gold_seg)
        pred_tags, gold_tags = align_seqs(pred_tags, gold_tags)
        pred_comb, gold_comb = align_seqs(pred_comb, gold_comb)

        all_gold_seg.extend(gold_seg)
        all_pred_seg.extend(pred_seg)

        all_gold_tag.extend(gold_tags)
        all_pred_tag.extend(pred_tags)

        all_gold.extend(gold_comb)
        all_pred.extend(pred_comb)

    f1_micro = f1_score(all_gold_tag, all_pred_tag, average="micro")
    f1_macro = f1_score(all_gold_tag, all_pred_tag, average="macro")
    print(f"{lang}:")
    print(f"Classification micro F1: {f1_micro:.4f}, macro f1: {f1_macro:.4f}")
    # print(classification_report(all_gold_tag, all_pred_tag))

    f1_micro = f1_score(all_gold_seg, all_pred_seg, average="micro")
    f1_macro = f1_score(all_gold_seg, all_pred_seg, average="macro")
    print(f"Segmentation micro F1: {f1_micro:.4f}, macro f1: {f1_macro:.4f}")

    f1_micro = f1_score(all_gold, all_pred, average="micro")
    f1_macro = f1_score(all_gold, all_pred, average="macro")
    print(f"Joint micro F1: {f1_micro:.4f}, macro f1: {f1_macro:.4f}")

    print(f"Joint accuracy: {correct / total:.4f}")

    print()
