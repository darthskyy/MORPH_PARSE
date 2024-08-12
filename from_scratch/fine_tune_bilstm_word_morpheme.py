import datetime

from ray import tune
from ray.util.client import ray

from lstm import BiLSTMTagger
from common import AnnotatedCorpusDataset, split_words, tokenize_into_chars, EmbedBySumming, tune_model, train_model, \
    model_for_config, tokenize_into_morphemes, EmbedSingletonFeature

model = (
    "bilstm",
    lambda train_set, embed, config: BiLSTMTagger(embed, config, train_set)
)
splits = (split_words, "word", 20)
feature_level = (
    "morpheme",
    {
        "embed_target_embed": tune.grid_search([128, 256]),
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
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-10, 1e-5),
        "hidden_dim": tune.grid_search([128, 256, 512]),
        "dropout": tune.grid_search([0.1, 0.2, 0.3]),
        "batch_size": tune.choice([1]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.grid_search([1, 2, 4]),
    }

    train, valid = AnnotatedCorpusDataset.load_data("ZU", split=split, tokenize=extract_features, use_surface=True)
    tune_model(model, cfg, feature_level, name, epochs, train, valid)


def final_train():
    cfg = {
      'lr': 0.0002948382869797967,
      'weight_decay': 0,
      'hidden_dim': 256,
      'dropout': 0.2,
      'batch_size': 1,
      'epochs': 20,
      'gradient_clip': 2,
      'embed_target_embed': 128
    }

    for lang in ["ZU"]:
        print(f"Training {split_name}-level, {feature_name}-feature {model_name} for {lang}")
        train, valid = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features, use_testset=False, use_surface=True)
        train_model(
            model_for_config(mk_model, embed_features, train, cfg), f"{model_name}-{lang}", cfg, train,
            valid, use_ray=False
        )


fine_tune()
print("Done at", datetime.datetime.now())
ray.shutdown()
