import datetime

from ray import tune
from ray.util.client import ray

from lstm import BiLSTMTagger
from common import AnnotatedCorpusDataset, split_words, tokenize_into_chars, EmbedBySumming, tune_model, EmbedWithBiLSTM


def main():
    model = (
        "bilstm",
        lambda train_set, embed, config: BiLSTMTagger(embed, config, train_set)
    )
    splits = (split_words, "word", 20)
    feature_level = (
        "character-bilstm",
        {
            "embed_hidden_embed": tune.choice([128]),
            "embed_hidden_dim": tune.choice([64]),
            "embed_target_embed": tune.choice([256]),
        },
        tokenize_into_chars,
        lambda config, dset: EmbedWithBiLSTM(dset, config["embed_hidden_embed"], config["embed_hidden_dim"],
                                             config["embed_target_embed"]),
    )
    lang = "ZU"

    split, split_name, epochs = splits
    model_name, mk_model = model
    (feature_name, _, extract_features, embed_features) = feature_level
    print(f"Tuning {split_name}-level, {feature_name}-feature {model_name} for {lang}")
    train, valid = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features)
    name = f"split-{split_name}feature-{feature_level}_model-{model_name}"

    cfg = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": 0,
        "hidden_dim": tune.grid_search([128, 256, 512]),
        "dropout": tune.grid_search([0, 0.1, 0.2]),
        "batch_size": tune.grid_search([1, 2, 4]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.grid_search([0.5, 1, 2, 4, float("inf")]),
    }

    tune_model(model, cfg, feature_level, name, epochs, train, valid, cpus=2)


main()
print("Done at", datetime.datetime.now())
ray.shutdown()
