import datetime

from ray import tune
from ray.util.client import ray

from bilstm_crf import BiLstmCrfTagger
from common import AnnotatedCorpusDataset, split_words, tune_model, train_model, \
    model_for_config, tokenize_into_morphemes, EmbedSingletonFeature, split_sentences

model = (
    "bilstm-crf",
    lambda train_set, embed, config: BiLstmCrfTagger(embed, config, train_set)
)
splits = (split_sentences, "sentences", 20)
feature_level = (
    "morpheme",
    {
        "embed_target_embed": tune.grid_search([256]),
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
        "lr": tune.loguniform(1e-6, 1e-2),
        "weight_decay": tune.choice([0.0]),
        "hidden_dim": tune.choice([1024]),
        "dropout": tune.choice([0.2]),
        "batch_size": tune.choice([2]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.choice([2])
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