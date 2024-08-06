import datetime

from ray import tune
from ray.util.client import ray

from bilstm_crf import BiLstmCrfTagger
from common import AnnotatedCorpusDataset, split_words, tune_model, train_model, \
    model_for_config, tokenize_into_morphemes, EmbedSingletonFeature, split_sentences, split_sentences_embedded_sep, \
    EmbedBySumming, tokenize_into_chars, EmbedWithBiLSTM

model = (
    "bilstm-crf",
    lambda train_set, embed, config: BiLstmCrfTagger(embed, config, train_set)
)
splits = (split_sentences_embedded_sep, "sentences_embedded_sep", 20)
feature_level = (
    "character-bilstm",
    {
        "embed_hidden_embed": tune.grid_search([2 ** i for i in range(7, 10)][::-1]),
        "embed_hidden_dim": tune.grid_search([2 ** i for i in range(7, 10)][::-1]),
        "embed_target_embed": tune.grid_search([2 ** i for i in range(7, 10)][::-1]),
    },
    tokenize_into_chars,
    lambda config, dset: EmbedWithBiLSTM(dset, config["embed_hidden_embed"], config["embed_hidden_dim"],
                                             config["embed_target_embed"]),
)

split, split_name, epochs = splits
model_name, mk_model = model
(feature_name, _, extract_features, embed_features) = feature_level

name = f"split-{split_name}feature-{feature_level}_model-{model_name}"


def fine_tune():
    print(f"Tuning {split_name}-level, {feature_name}-feature {model_name} for ZU")
    cfg = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-10, 1e-5),
        "hidden_dim": tune.grid_search([128, 256, 512]),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
        "batch_size": tune.choice([1, 2, 4, 8, 16]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.choice([0.5, 1, 2, 4, float("inf")])
    }

    train, valid = AnnotatedCorpusDataset.load_data("ZU", split=split, tokenize=extract_features)
    tune_model(model, cfg, feature_level, name, epochs, train, valid)


def final_train():
    cfg = {}  # TODO

    for lang in ["XH", "ZU", "SS", "NR"]:
        print(f"Training {split_name}-level, {feature_name}-feature {model_name} for {lang}")
        train, valid = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features, use_testset=True)
        train_model(
            model_for_config(mk_model, embed_features, train, cfg), f"{model_name}-{lang}", cfg, train,
            valid, use_ray=False
        )


fine_tune()

print("Done at", datetime.datetime.now())
ray.shutdown()
