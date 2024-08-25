import os
import pickle
import random
from pathlib import Path

import torch
import datetime
from ray import tune
from ray.util.client import ray

from from_scratch.dataset import split_sentences_raw, prepare_sequence, SEQ_PAD_TEXT, WORD_SEP_TEXT, SEQ_PAD_IX, \
    WORD_SEP_IX, UNK_IDX
from from_scratch.transformer import TransformerModel
from common import AnnotatedCorpusDataset, train_model, split_words, tokenize_into_morphemes, \
    tokenize_into_chars, split_sentences, EmbedBySumming, EmbedSingletonFeature, \
    EmbedWithBiLSTM, analyse_model, tune_model, model_for_config, EmbedRawChars, train_all

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = (
    "transformer",
    lambda train_set, embed, config: TransformerModel(embed, config, len(train_set.tag_to_ix))
)
splits = (split_words, "words", 20)
feature_level = (
    "character-raw",
    {
        "embed_target_embed": tune.grid_search([256, 512, 1024][::-1]),
    },
    tokenize_into_chars,
    lambda config, dset: EmbedRawChars(dset, config["embed_target_embed"])
)


split, split_name, epochs = splits
model_name, mk_model = model
(feature_name, _, extract_features, embed_features) = feature_level

name = f"split-{split_name}_feature-{feature_level}_model-{model_name}"


class TransformerDataset:
    def __init__(self, sentences, tag_to_ix, ix_to_tag, char_to_ix, ix_to_char):
        self.sentences = sentences

        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = ix_to_tag
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char

        self.ix_to_morpheme = ix_to_char
        self.num_submorphemes = len(self.char_to_ix)
        self.is_surface = False


    @staticmethod
    def load_data(lang):
        sentences = list(split_sentences_raw(TransformerDataset.read_lines(lang)))

        SOS = "<?SOS?>"
        EOS = "<?EOS?>"

        char_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: WORD_SEP_IX, "<?unk?>": UNK_IDX, SOS: UNK_IDX + 1, EOS: UNK_IDX + 2}
        ix_to_char = {SEQ_PAD_IX: SEQ_PAD_TEXT, WORD_SEP_IX: WORD_SEP_TEXT, UNK_IDX: "<?unk?>", UNK_IDX + 1: SOS, UNK_IDX + 2: EOS}

        tag_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: WORD_SEP_IX, EOS: WORD_SEP_IX + 1}
        ix_to_tag = {SEQ_PAD_IX: SEQ_PAD_TEXT, WORD_SEP_IX: WORD_SEP_TEXT, WORD_SEP_IX + 1: EOS}

        for words, tags in sentences:
            for tag in tags:
                if tag not in tag_to_ix:
                    ix = len(tag_to_ix)
                    tag_to_ix[tag] = ix
                    ix_to_tag[ix] = tag

        test_amount = len(sentences) // 10
        random.shuffle(sentences)

        test_sentences = sentences[:test_amount]
        train_sentences = sentences[test_amount:]

        for words, tags in train_sentences:
            for char in words:
                if char not in char_to_ix:
                    ix = len(char_to_ix)
                    char_to_ix[char] = ix
                    ix_to_char[ix] = char

        def prepare_dset(all_sentences):
            return [
                (prepare_sequence(chars, char_to_ix), prepare_sequence(tag_seq, tag_to_ix))
                for (chars, tag_seq) in all_sentences
            ]
        print("tags", len(ix_to_tag))

        return (
            TransformerDataset(prepare_dset(train_sentences), tag_to_ix, ix_to_tag, char_to_ix, ix_to_char),
            TransformerDataset(prepare_dset(test_sentences), tag_to_ix, ix_to_tag, char_to_ix, ix_to_char),
        )

    def to(self, dev):
        self.sentences = [(a.to(dev), b.to(dev)) for a, b in self.sentences]

    def __getitem__(self, item):
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)



    @staticmethod
    def read_lines(lang):
        with open(f"data/TRAIN/{lang}_TRAIN.tsv") as f:
            for line in f.readlines():
                cols = line.strip().split("\t")
                word = cols[0]
                tag_seq = cols[3].split("_")
                yield (word, tag_seq)


def fine_tune():
    torch.manual_seed(0)
    random.seed(0)

    print(f"Tuning {split_name}-level, {feature_name}-feature {model_name} for ZU")
    cfg = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-10, 1e-5),
        "hidden_dim": tune.grid_search([64, 128, 256]),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
        "batch_size": tune.choice([1, 2, 4]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.choice([0.5, 1, 2])
    }

    train, valid = AnnotatedCorpusDataset.load_data("ZU", split=split, tokenize=extract_features)
    tune_model(model, cfg, feature_level, name, epochs, train, valid)


def final_train():
    torch.manual_seed(0)
    random.seed(0)

    cfg = {
        'hidden_dim': 1024, # TODO try different hidden dim here too
        'embed_target_embed': 64,
        'lr': 2e-3,
        'weight_decay': 0,
        'dropout': 0.1,
        'batch_size': 8,
        'epochs': 40,
        'gradient_clip': float('inf'),
        'layers': 2,
        'heads': 8
    }

    train_all(model, splits, feature_level, cfg)


final_train()
print("Done at", datetime.datetime.now())
ray.shutdown()
