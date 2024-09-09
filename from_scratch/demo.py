import re

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from from_scratch.dataset import (split_sentences_raw, extract_morphemes_and_tags_from_file_2022, WORD_SEP_TEXT,
                                  SEQ_PAD_TEXT, identity, tags_only_no_classes, classes_only_no_tags)
from from_scratch.encapsulated_model import EncapsulatedModel
from aligned_f1 import align_seqs

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tag_pattern = re.compile(r'\[[a-zA-Z-_0-9|]*?]-?')


def segment_query(q):
    if tag_pattern.search(q):
        return [[morpheme for morpheme in tag_pattern.split(word) if morpheme != ""] for word in q.split(" ")]
    else:
        return [word.split("-") for word in q.split(" ")]


def split_words(seq):
    word = []
    for i in seq:
        if i != WORD_SEP_TEXT:
            word.append(i)
        else:
            yield word
            word = []
    if word:
        yield word


def eval_model(model, test_set, map_tag=identity):
    gold, pred = [], []
    out = []
    for morphemes, gold_tags in test_set:
        if not morphemes:
            continue

        gold_tags_per_word = list(split_words(gold_tags))
        morphemes_per_word = list(split_words(morphemes))

        for pred_word, gold_word in zip(model.forward(morphemes_per_word), gold_tags_per_word):
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


def demo():
    while True:
        # path = input("Model > ")
        # for lang in ("ZU",):
        for lang in ("ZU", "NR", "XH", "SS"):
            path = f"crf_sentence-embed_morpheme_canon/bilstm-crf-sentences_embedded_sep-morpheme-{lang}.pt"
            model: EncapsulatedModel = torch.load("out_models/" + path, map_location=device)
            model.eval()

            with torch.no_grad():
                model_parameters = filter(lambda p: p.requires_grad, model.parameters())
                params = sum([np.prod(p.size()) for p in model_parameters])
                print(f"Model has {params / 1_000_000:.2f}M parameters")  # TODO

                # suffix = "_SURFACE" if model.is_surface else ""
                suffix = "_CANONICAL_PRED"
                path = f"data/TEST/{model.lang}_TEST{suffix}.tsv"
                sentences = list(split_sentences_raw(extract_morphemes_and_tags_from_file_2022(path, use_surface=model.is_surface, is_demo=True)))

                print("====== Full tagset (syntactic & noun class) ======")  # TODO
                micro, macro, report = eval_model(model, sentences)
                print(report)  # TODO
                print(f"{lang} Micro F1: {micro:.10f} ({micro:.4f}). Macro F1: {macro:.10f} ({macro:.4f})")

                # print("====== Syntactic tagset only ======")
                # micro, macro, report = eval_model(model, sentences, map_tag=tags_only_no_classes)
                # print(report)
                # print(f"Syntactic Micro F1: {micro:.10f} ({micro:.4f}). Macro F1: {macro:.10f} ({macro:.4f})")

                continue  # TODO

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