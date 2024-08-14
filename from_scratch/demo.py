from multiprocessing import Pool

import torch
from sklearn.metrics import f1_score, classification_report

from dataset import split_sentences_raw, extract_morphemes_and_tags_from_file_2022, WORD_SEP_TEXT, \
    tokenize_into_lower_morphemes
from from_scratch.encapsulated_model import EncapsulatedModel
from aligned_f1 import align_seqs

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def split_words(seq):
    word = []
    for i in seq:
        if i != WORD_SEP_TEXT:
            word.append(i)
        else:
            yield word
            word = []
    return word


while True:
    # path = input("Model > ")
    path = "bilstm-word-ZU.pt"
    model: EncapsulatedModel = torch.load("out_models/" + path, map_location=device)

    # TODO
    model.tokenize = tokenize_into_lower_morphemes

    suffix = "_SURFACE" if model.is_surface else ""
    path = f"data/TEST/{model.lang}_TEST{suffix}.tsv"
    sentences = split_sentences_raw(extract_morphemes_and_tags_from_file_2022(path, use_surface=model.is_surface))

    print("Getting preds")
    gold, pred = [], []
    out = []
    for morphemes, gold_tags in sentences:
        if not morphemes:
            continue

        gold_tags_per_word = list(split_words(gold_tags))
        morphemes_per_word = list(split_words(morphemes))

        for pred_word, gold_word in zip(model.forward(morphemes_per_word), gold_tags_per_word):
            pred_word, gold_word = align_seqs(pred_word, gold_word)

            pred.extend(pred_word)
            gold.extend(gold_word)

    # print("thread pooling")
    # pool = Pool()
    # out = pool.map(process, out)
    # pool.close()
    #
    # gold, pred = [], []
    # for (gold_word, pred_word) in out:
    #     gold.extend(gold_word)
    #     pred.append(pred_word)

    print(pred[:100])
    print(gold[:100])

    micro = f1_score(gold, pred, average="micro")
    macro = f1_score(gold, pred, average="macro")
    report = classification_report(gold, pred, zero_division=0.0)

    print(report)
    print(f"Micro F1: {micro:.4f}. Macro F1: {macro:.4f}")

    while True:
        query = input("Morpological segmentation (separated by _) or Q to quit >")
        # query = "nga-tulu kwa-loko , ku-b-a khona ku-niket-el-a"
        if query.lower().strip() == "q":
            break

        words = [word.split("_") for word in query.split(" ")]
        print(model.forward(words))
