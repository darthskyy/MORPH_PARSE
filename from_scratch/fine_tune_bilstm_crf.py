import torch
import datetime
from ray import tune
from ray.util.client import ray

from bilstm_crf import BiLstmCrfTagger
from common import AnnotatedCorpusDataset, split_words, tokenize_into_morphemes, EmbedSingletonFeature, train_all, \
    tune_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = (
    "bilstm-crf",
    lambda train_set, embed, config: BiLstmCrfTagger(embed, config, train_set)
)
splits = (split_words, "word", 20)
feature_level = (
    "morpheme",
    {
        "embed_target_embed": tune.choice([256]),
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
        'lr': 0.0001723733700380886,
        'weight_decay': 2.6412896449908767e-07,
        'hidden_dim': 1024,
        'dropout': 0.2,
        'batch_size': 1,
        'epochs': 40,
        'gradient_clip': 4,
        'embed_target_embed': 512
    }

    train_all(model, splits, feature_level, cfg)


final_train()
print("Done at", datetime.datetime.now())
ray.shutdown()
