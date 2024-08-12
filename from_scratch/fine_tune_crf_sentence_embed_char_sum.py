import datetime

from ray import tune
from ray.util.client import ray

from bilstm_crf import BiLstmCrfTagger
from common import AnnotatedCorpusDataset, split_words, tune_model, train_model, \
    model_for_config, tokenize_into_morphemes, EmbedSingletonFeature, split_sentences, split_sentences_embedded_sep, tokenize_into_chars, EmbedBySumming

model = (
    "bilstm-crf",
    lambda train_set, embed, config: BiLstmCrfTagger(embed, config, train_set)
)
splits = (split_sentences_embedded_sep, "sentences_embedded_sep", 20)
feature_level = (
    "character-summing",
    {
     	"embed_target_embed": tune.choice([128, 256, 512])
    },
    tokenize_into_chars,
    lambda config, dset: EmbedBySumming(dset, config["embed_target_embed"])
)

split, split_name, epochs = splits
model_name, mk_model = model
(feature_name, _, extract_features, embed_features) = feature_level

name = f"split-{split_name}feature-{feature_level}_model-{model_name}"


def fine_tune():
    print(f"Tuning {split_name}-level, {feature_name}-feature {model_name} for ZU")
    cfg = {
        "lr": tune.loguniform(1e-6, 1e-3),
        "weight_decay": tune.loguniform(1e-11, 1e-7),
        "hidden_dim": tune.choice([56, 512, 1024]),
        "dropout": tune.grid_search([0.0, 0.1, 0.2, 0.3]),
        "batch_size": tune.grid_search([1, 2, 4, 8]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.choice([0.5, 1, 2])
    }

    train, valid = AnnotatedCorpusDataset.load_data("ZU", split=split, tokenize=extract_features)
    tune_model(model, cfg, feature_level, name, epochs, train, valid)


def final_train():
    for lang in ["XH", "ZU", "SS", "NR"]:
        cfg = {
            "lr": 0.0002070522419581859,
            "weight_decay": 0,
            "hidden_dim": 512,
            "dropout": 0.2,
            "batch_size": 2,
            "epochs": 20,
            "gradient_clip": 1,
            "embed_target_embed": 512
        }

        print(f"Training {split_name}-level, {feature_name}-feature {model_name} for {lang}")
        train, valid = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features, use_testset=True)
        train_model(
            model_for_config(mk_model, embed_features, train, cfg), f"{model_name}-{lang}", cfg, train,
            valid, use_ray=False
        )


fine_tune()

print("Done at", datetime.datetime.now())
ray.shutdown()