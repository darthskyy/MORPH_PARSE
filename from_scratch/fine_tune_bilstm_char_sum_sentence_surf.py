import datetime

from ray import tune
from ray.util.client import ray

from bilstm_crf import BiLstmCrfTagger
from common import (AnnotatedCorpusDataset, tune_model, tokenize_into_morphemes, EmbedSingletonFeature, split_sentences,
                    train_all, EmbedBySumming, tokenize_into_chars)
from dataset import tokenize_into_lower_chars
from lstm import BiLSTMTagger

# Some configurable aspects of the model - the model itself, context level, and submorpheme tokenisation
model = (
    "bilstm",
    lambda train_set, embed, config: BiLSTMTagger(embed, config, train_set)
)
splits = (split_sentences, "sentences", 20)
feature_level = (
    "character-summing-lower",
    {
        "embed_target_embed": tune.choice([64, 128, 256])
    },
    tokenize_into_lower_chars,
    lambda config, dset: EmbedBySumming(dset, config["embed_target_embed"])
)

split, split_name, epochs = splits
model_name, mk_model = model
(feature_name, _, extract_features, embed_features) = feature_level

name = f"split-{split_name}feature-{feature_level}_model-{model_name}"


def fine_tune():
    """Tune the model to select best hyperparameters"""

    print(f"Tuning {split_name}-level, {feature_name}-feature {model_name} for ZU")
    cfg = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-10, 1e-5),
        "hidden_dim": tune.grid_search([256, 512, 1024][::-1]),
        "dropout": tune.choice([0.1, 0.2]),
        "batch_size": tune.choice([4]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.choice([0.5, 1, 2, 4, float("inf")])
    }

    train, valid = AnnotatedCorpusDataset.load_data("ZU", split=split, tokenize=extract_features)
    tune_model(model, cfg, feature_level, name, epochs, train, valid)


def final_train():
    """Train & save the model with a given config"""

    cfg = {
        'lr': 0.0026403282274115964,
        'weight_decay': 0,
        'hidden_dim': 1024,
        'dropout': 0.2,
        'batch_size': 1,
        'epochs': {
            "NR": 33,
            "SS": 32,
            "XH": 31,
            "ZU": 31,
        },
        'gradient_clip': 2,
        'embed_target_embed': 128,
    }

    train_all(model, splits, feature_level, cfg, use_testset=True, use_surface=True)


final_train()

print("Done at", datetime.datetime.now())
ray.shutdown()
