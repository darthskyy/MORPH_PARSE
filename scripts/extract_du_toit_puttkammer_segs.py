import csv

from sklearn.metrics import f1_score

from aligned_f1 import align_seqs
from data_prep import split_tags
from from_scratch.dataset import tags_only_no_classes

for lang in ("ZU", "NR", "XH", "SS"):
    with open(f"data/DuToitPuttkammer/MorphAnalysis.{lang}.{lang}_TEST_OUT.txt") as f:
        pred_lines = [line.split("\t") for line in f.read().splitlines(keepends=False) if line]

    with open(f"data/TEST/{lang}_TEST.tsv") as f:
        gold_lines = [line.split("\t") for line in f.read().splitlines(keepends=False) if line]

    with open(f"data/TEST/{lang}_TEST_DTP_CANONICAL_PRED.tsv", "w") as f:
        for pred_line, gold_line in zip(pred_lines, gold_lines):
            pred_raw, pred_analysis = pred_line
            gold_raw, _, _, gold_tags = gold_line
            assert pred_raw.lower() == gold_raw.lower()
            pred_seg, pred_tags = split_tags(pred_analysis)

            f.write("\t".join([gold_raw, "_", "_".join(pred_seg), gold_tags]) + "\n")

