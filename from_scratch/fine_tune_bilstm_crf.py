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
    EmbedWithBiLSTM, analyse_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

models = [
    (
        "bilstm-crf",
        lambda train_set, embed, config: BiLstmCrfTagger(embed, config, train_set)
    ),
    # (
    #     "bilstm",
    #     lambda train_set, embed, config: BiLSTMTagger(embed, config, train_set)
    # ),
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
    # ("character-summing", tokenize_into_chars, lambda dset: EmbedBySumming(dset, 512)),
    (
        "morpheme",
        {
            "embed_target_embed": tune.choice([256]),
        },
        tokenize_into_morphemes,
        lambda config, dset: EmbedSingletonFeature(dset, config["embed_target_embed"])
    ),
]

results_by_lang = dict()


def tune_model(model, feature_level, name: str, epochs, train: AnnotatedCorpusDataset,
               valid: AnnotatedCorpusDataset):
    embed_config, mk_embed = feature_level[1], feature_level[3]
    name, mk_model = model[0], model[1]

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-10, 1e-5),
        "hidden_dim": tune.choice([256, 512]),
        "dropout": tune.choice([0.1]),
        "batch_size": tune.choice([1]),
        "epochs": tune.choice([epochs]),
        "gradient_clip": tune.choice([2]),
        **embed_config
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        grace_period=3,
        reduction_factor=2,
    )

    train, valid = ray.put(train), ray.put(valid)
    result = tune.run(
        lambda conf: train_model(model_for_config(mk_model, mk_embed, ray.get(train), conf), name, conf, ray.get(train), ray.get(valid)),
        resources_per_trial={"gpu": 0.25} if torch.cuda.is_available() else None,
        config=config,
        num_samples=100,
        time_budget_s=11 * 60 * 60,  # 11h
        search_alg=BasicVariantGenerator(constant_grid_search=True, max_concurrent=4),
        scheduler=scheduler,
        storage_path=os.environ["TUNING_CHECKPOINT_DIR"],
    )

    for metric in ["f1_macro", "f1_micro"]:
        best_trial = result.get_best_trial(metric, "max", "all")
        print(f"Best trial by {metric}:")
        print(f" config: {best_trial.config}")
        print(f" val loss: {best_trial.last_result['loss']}")
        print(f" macro fs1 {best_trial.last_result['f1_macro']}")
        print(f" micro {best_trial.last_result['f1_micro']}")

        best_model = model_for_config(mk_model, mk_embed, ray.get(train), best_trial.config)
        best_model.to(device)

        best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric=metric, mode="max")
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                best_checkpoint_data = pickle.load(fp)

            best_model.load_state_dict(best_checkpoint_data["model_state_dict"])
            _, _, report, f1_micro, f1_macro, f1_weighted = analyse_model(best_model, best_trial.config, ray.get(valid))
            print(f" {name}: Micro F1: {f1_micro}. Macro f1: {f1_macro}. Weighted F1: {f1_weighted}")
            print(f" {name}: Best macro f1: {best_checkpoint_data['best_macro']} at epoch {best_checkpoint_data['best_epoch']}")
            print(report)

    # return f1_micro, f1_weighted, f1_macro


def model_for_config(mk_model, mk_embed, train, config):
    embed_module = mk_embed(config, train).to(device)
    model = mk_model(train, embed_module, config).to(device)
    return model


def main():
    ray.init(num_cpus=4)
    for model in models:
        for (split, split_name, epochs) in splits:
            for feature_level in feature_levels:
                for lang in ["ZU"]:
                    model_name, mk_model = model
                    (feature_name, _, extract_features, embed_features) = feature_level
                    print(f"Tuning {split_name}-level, {feature_name}-feature {model_name} for {lang}")
                    train, valid = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features)
                    name = f"split-{split_name}feature-{feature_level}_model-{model_name}"
                    tune_model(model, feature_level, name, epochs, train, valid)


def final_train():
    cfg = {
        'lr': 0.0003253641575685023,
        'weight_decay': 1.711945898815254e-06,
        'hidden_dim': 256,
        'dropout': 0.1,
        'batch_size': 16,
        'epochs': 30,
        'gradient_clip': 1,
        'embed_target_embed': 128
    }

    for model in models:
        for (split, split_name, epochs) in splits:
            for feature_level in feature_levels:
                for lang in ["ZU", "XH", "NR", "SS"]:
                    model_name, mk_model = model
                    (feature_name, _, extract_features, embed_features) = feature_level
                    print(f"Training {split_name}-level, {feature_name}-feature {model_name} for {lang}")

                    train, valid = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features, use_testset=True)
                    train_model(
                        model_for_config(mk_model, embed_features, train, cfg), f"{model_name}-{lang}", cfg, train,
                        valid, use_ray=False
                    )


final_train()
# main()
print("Done at", datetime.datetime.now())
ray.shutdown()

