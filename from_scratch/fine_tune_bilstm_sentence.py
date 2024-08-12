import os
import pickle
from pathlib import Path

import torch
import datetime
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator
from ray.util.client import ray

from lstm import BiLSTMTagger
from bilstm_crf import BiLstmCrfTagger
from common import AnnotatedCorpusDataset, train_model, split_words, tokenize_into_morphemes, \
    tokenize_into_chars, split_sentences, split_sentences_no_sep, EmbedBySumming, EmbedSingletonFeature, \
    EmbedWithBiLSTM, analyse_model, split_sentences_embedded_sep, tune_model, model_for_config, tags_only_no_classes, \
    classes_only_no_tags, tokenize_into_trigrams_with_sentinels, identity, tokenize_into_trigrams_no_sentinels, \
    tokenize_into_lower_morphemes

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = (
    "bilstm",
    lambda train_set, embed, config: BiLSTMTagger(embed, config, train_set)
)
splits = (split_sentences, "sentences", 20)
# splits = (split_sentences_embedded_sep, "sentences_embedded_sep", 20)
# splits = (split_words, "words", 20)
# feature_level = (
#     "trigram-sum",
#     {
#         "embed_target_embed": tune.grid_search([128, 256, 512]),
#     },
#     tokenize_into_trigrams_no_sentinels,
#     lambda config, dset: EmbedBySumming(dset, config["embed_target_embed"])
# )

feature_level = (
    "morpheme",
    {
        "embed_target_embed": tune.grid_search([128, 256, 512]),
    },
    tokenize_into_morphemes,
    lambda config, dset: EmbedSingletonFeature(dset, config["embed_target_embed"])
)

split, split_name, epochs = splits
model_name, mk_model = model
(feature_name, _, extract_features, embed_features) = feature_level

name = f"split-{split_name}feature-{feature_level}_model-{model_name}"


def fine_tune():
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

    train, valid = AnnotatedCorpusDataset.load_data("ZU", split=split, tokenize=extract_features, use_surface=True)
    tune_model(model, cfg, feature_level, name, epochs, train, valid)


def final_train():
    cfg = {
        'lr': 0.00019401697177446437,
        'weight_decay': 9.230833759168172e-07,
        'hidden_dim': 128,
        'dropout': 0.1,
        'batch_size': 1,
        'epochs': 20,
        'gradient_clip': 1,
        'embed_target_embed': 256
    }
    #
    # cfg = {
    #     'lr': 0.0007301058410653952,
    #     'weight_decay': 1.5329333022359804e-07,
    #     'hidden_dim': 64,
    #     'dropout': 0.1,
    #     'batch_size': 1,
    #     'epochs': 20,
    #     'gradient_clip': 1,
    #     'embed_target_embed': 128
    # }

    for lang in ["ZU"]:
        train, valid = AnnotatedCorpusDataset.load_data("ZU", split=split, tokenize=extract_features, use_testset=False, use_surface=True)
        macros = []
        for seed in [0, 12904, 1028485, 2795]:
            print(f"Training {split_name}-level, {feature_name}-feature {model_name} for {lang}")
            torch.manual_seed(seed)
            _, macro, _ = train_model(
                model_for_config(mk_model, embed_features, train, cfg), f"{model_name}-{lang}", cfg, train,
                valid, use_ray=False
            )
            macros.append(macro)
        print("Average across 4 seeds:", float(sum(macros)) / 4.0)


fine_tune()
print("Done at", datetime.datetime.now())
ray.shutdown()