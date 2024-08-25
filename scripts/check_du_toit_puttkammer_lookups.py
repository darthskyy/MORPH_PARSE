import csv

from sklearn.metrics import f1_score, classification_report

from aligned_f1 import align_seqs
from data_prep import split_tags
from from_scratch.dataset import tags_only_no_classes

# for lang in ("ZU", "NR", "XH", "SS"):
for lang in (("NR", "XH", "SS", "ZU")):
    with open(f"data/TEST/{lang}_TEST.tsv") as f:
        test_words = set([line.split("\t")[0].lower() for line in f.read().splitlines(keepends=False) if line])

    with open(f"data/TRAIN/{lang}_TRAIN.tsv") as f:
        train_words = set([line.split("\t")[0].lower() for line in f.read().splitlines(keepends=False) if line])

    with open(f"data/DuToitPuttkammer/{lang.lower()}-lookup.txt") as f:
        lookup_words = [line.split("\t")[0].lower() for line in f.read().splitlines(keepends=False) if line]

    only_in_test, only_in_train, in_both, total = 0, 0, 0, 0
    for word in lookup_words:
        if word in train_words and word not in test_words:
            only_in_train += 1
        if word in test_words and word not in train_words:
            only_in_test += 1
            print(word)
        total += 1

    print(f"{lang} lookups: Out of {total} words, {only_in_train} ({only_in_train / total * 100:.2f}%) only in train and {only_in_test} ({only_in_test / total * 100:.2f}%) only in test")

    only_in_test, total = 0, 0
    for word in test_words:
        if word not in train_words:
            only_in_test += 1
        total += 1
    print(f"{lang} test: Out of {total} words, {only_in_test} ({only_in_test / total * 100:.2f}%) only in test")
