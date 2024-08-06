import os
import pickle
import sys
from pathlib import Path

import torch
import datetime
from ray import tune
from ray.util.client import ray

from lstm import BiLSTMTagger
from bilstm_crf import BiLstmCrfTagger
from common import AnnotatedCorpusDataset, train_model, split_words, tokenize_into_morphemes, \
    tokenize_into_chars, split_sentences, split_sentences_no_sep, EmbedBySumming, EmbedSingletonFeature, \
    EmbedWithBiLSTM, analyse_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

models = [
    # (
    #     "bilstm-crf",
    #     lambda train_set, embed, config: BiLstmCrfTagger(embed, config, train_set)
    # ),
    (
        "bilstm",
        lambda train_set, embed, config: BiLSTMTagger(embed, config, train_set)
    ),
]

splits = [
    # (split_sentences, "sentence", 20),
    (split_words, "word", 20),
]

feature_levels = [
    # (
    #     "character-bilstm",
    #     tokenize_into_chars,
    #     {
    #         "embed_hidden_embed": tune.choice([2 ** i for i in range(2, 10)]),
    #         "embed_hidden_dim": tune.choice([2 ** i for i in range(3, 10)]),
    #         "embed_target_embed": tune.choice([2 ** i for i in range(3, 10)]),
    #     },
    #     lambda config, dset: EmbedWithBiLSTM(dset, config["embed_hidden_embed"], config["embed_hidden_dim"],
    #                                          config["embed_target_embed"]),
    # ),
    (
        "character-summing",
        {
            "embed_target_embed": tune.grid_search([64, 128, 256, 512][::-1])
        },
        tokenize_into_chars,
        lambda config, dset: EmbedBySumming(dset, config["embed_target_embed"])
    )
    # (
    #     "morpheme",
    #     {
    #         "embed_target_embed": tune.choice([128, 256, 512]),
    #     },
    #     tokenize_into_morphemes,
    #     lambda config, dset: EmbedSingletonFeature(dset, config["embed_target_embed"])
    # ),
]

results_by_lang = dict()


def model_for_config(mk_model, mk_embed, train, config):
    embed_module = mk_embed(config, train).to(device)
    model = mk_model(train, embed_module, config).to(device)
    return model


def main():
    for model in models:
        for (split, split_name, epochs) in splits:
            for feature_level in feature_levels:
                for lang in ["ZU", "NR", "XH", "SS"]:
                    model_name, mk_model = model
                    (feature_name, _, extract_features, embed_features) = feature_level
                    print(f"Tuning {split_name}-level, {feature_name}-feature {model_name} for {lang}")
                    train, valid = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features)
                    name = f"split-{split_name}_feature-{feature_name}_model-{model_name}"

                    conf = {
                        'embed_target_embed': 256,
                        'lr': 0.00012620771127627717,
                        'weight_decay': 2.8931903058581684e-06,
                        'hidden_dim': 256,
                        'dropout': 0,
                        'batch_size': 1,
                        'epochs': 20,
                        'gradient_clip': 2,
                    }

                    train_model(
                        model_for_config(mk_model, embed_features, train, conf), name, conf, train,
                        valid, use_ray=False
                    )

                    # tune_model(model, feature_level, name, epochs, train, valid)
                    # micro, macro, weighted = tune_model(model, feature_level, name, epochs, train, valid)
                    # results_by_lang.setdefault(lang, [])
                    # results_by_lang[lang].append((name, micro, weighted, macro))

    # model_results = [f",{result[0]}," for result in results_by_lang["XH"]]
    # print(','.join(['Language', *model_results]))
    # print(',', ','.join('Micro, Weighted, Macro' for _ in results_by_lang["XH"]))
    # for lang in results_by_lang.keys():
    #     print(f"{lang},")
    #     for result in results_by_lang[lang]:
    #         print(','.join([f"{f1:.2f}" for f1 in result[1:]]))


main()
ray.shutdown()
