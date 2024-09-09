import datetime

from ray import tune
from ray.util.client import ray

from lstm import BiLSTMTagger
from common import AnnotatedCorpusDataset, split_words, tokenize_into_chars, EmbedBySumming, tune_model, train_model, \
    model_for_config, train_all, split_sentences, tokenize_into_morphemes, EmbedSingletonFeature

model = (
    "bilstm",
    lambda train_set, embed, config: BiLSTMTagger(embed, config, train_set)
)
splits = (split_sentences, "sentences", 20)
feature_level = (
    "morpheme",
    {
        "embed_target_embed": tune.choice([256, 512]),
    },
    tokenize_into_morphemes,
    lambda config, dset: EmbedSingletonFeature(dset, config["embed_target_embed"])
)

split, split_name, epochs = splits
model_name, mk_model = model
(feature_name, _, extract_features, embed_features) = feature_level

name = f"split-{split_name}feature-{feature_level}_model-{model_name}-surface"


def fine_tune():
    print(f"Tuning {split_name}-level, {feature_name}-feature {model_name} for ZU")
    cfg = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-10, 1e-5),
        "hidden_dim": tune.choice([128, 256, 512]),
        "dropout": tune.grid_search([0, 0.1]),
        "batch_size": tune.choice([8]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.grid_search([0.5, 1, 2, 4, float("inf")]),
    }

    train, valid = AnnotatedCorpusDataset.load_data("ZU", split=split, tokenize=extract_features, use_surface=True)
    tune_model(model, cfg, feature_level, name, epochs, train, valid)


def final_train():
    cfg = {
        'lr': 0.0005979875754024589,
        'weight_decay': 3.477922586045928e-08,
        'hidden_dim': 512,
        'dropout': 0,
        'batch_size': 1,
        'epochs': 40,
        'gradient_clip': 0.5,
        'embed_target_embed': 512
    }

    train_all(model, splits, feature_level, cfg, use_surface=True, use_testset=False)


final_train()
print("Done at", datetime.datetime.now())
ray.shutdown()
