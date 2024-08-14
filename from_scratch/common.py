import os
import pickle
import tempfile
import time
from pathlib import Path
import random

import torch
from torch import optim, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from sklearn.metrics import f1_score, classification_report
from ray.train import Checkpoint, get_checkpoint
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator
from ray.util.client import ray

from aligned_f1 import align_seqs
from encapsulated_model import EncapsulatedModel
from dataset import AnnotatedCorpusDataset, WORD_SEP_IX, SEQ_PAD_IX
import dataset

torch.manual_seed(0)
random.seed(0)

split_words = dataset.split_words
tokenize_into_chars = dataset.tokenize_into_chars
tokenize_into_morphemes = dataset.tokenize_into_morphemes
split_sentences = dataset.split_sentences

class EmbedBySumming(nn.Module):
    def __init__(self, trainset: AnnotatedCorpusDataset, target_embedding_dim):
        super(EmbedBySumming, self).__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.embed = nn.Embedding(trainset.num_submorphemes, target_embedding_dim, device=self.dev)
        self.output_dim = target_embedding_dim

    def forward(self, morphemes):
        return torch.sum(self.embed(morphemes), dim=2)


class EmbedSingletonFeature(nn.Module):
    def __init__(self, trainset: AnnotatedCorpusDataset, target_embedding_dim):
        super(EmbedSingletonFeature, self).__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.embed = nn.Embedding(trainset.num_submorphemes, target_embedding_dim, device=self.dev)
        self.output_dim = target_embedding_dim

    def forward(self, morphemes):
        assert morphemes.size(dim=2) == 1
        return torch.squeeze(self.embed(morphemes), dim=2)


class EmbedWithBiLSTM(nn.Module):
    def __init__(self, trainset: AnnotatedCorpusDataset, hidden_embed_dim, hidden_dim, target_embedding_dim,
                 num_layers=1,
                 dropout=0):
        super(EmbedWithBiLSTM, self).__init__()
        self.output_dim = target_embedding_dim
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.embed = nn.Embedding(trainset.num_submorphemes, hidden_embed_dim)
        self.lstm = nn.LSTM(hidden_embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True,
                            batch_first=True, device=self.dev)
        self.hidden2embed = nn.Linear(hidden_dim * 2, target_embedding_dim, device=self.dev)
        self.drop = nn.Dropout(dropout)  # Dropout used for input & output of bi-LSTM, as per NLAPOST21 paper

    def forward(self, batches):
        # We want to compute the combination of each word's subword representation
        # Therefore, we unbind on the batch level (each batch is a sentence / word), and then treat each morpheme as
        # a batch

        batches_out = []
        for morphemes in torch.unbind(batches, dim=0):
            embeds = self.embed(morphemes)
            embeds = self.drop(embeds)
            lstm_out, _ = self.lstm(embeds)
            lstm_out = self.drop(lstm_out)
            out_embeds = self.hidden2embed(lstm_out)
            batches_out.append(out_embeds[:, embeds.size(dim=1) - 1])

        return torch.stack(batches_out, dim=0)


def analyse_model(model, config, valid: AnnotatedCorpusDataset):
    valid_loader = DataLoader(
        valid,
        batch_sampler=BatchSampler(RandomSampler(valid), config["batch_size"], False),
        collate_fn=_collate_by_padding,
    )

    with torch.no_grad():
        # Set model to evaluation mode (affects layers such as BatchNorm)
        model.eval()

        predicted = []
        expected = []
        valid_loss = 0.0

        for morphemes, expected_tags in valid_loader:
            loss = model.loss(morphemes, expected_tags)
            valid_loss += loss.item() * morphemes.size(dim=0)

            predicted_tags = model.forward_tags_only(morphemes)

            # print("Sequence", list(valid.ix_to_morpheme[morph.item()] for morph in torch.flatten(morphemes)))
            # print("Expected", list(valid.ix_to_tag[tag.item()] for tag in torch.flatten(expected_tags)))
            # print("Predicted", list(valid.ix_to_tag[tag.item()] for tag in torch.flatten(predicted_tags)))
            # print()

            # This loop splits by batch
            for batch_elt_expected, batch_elt_pred in zip(torch.unbind(expected_tags), torch.unbind(predicted_tags)):
                # This loop splits by morpheme

                predicted_this_batch, expected_this_batch = [], []
                for expected_tag, predicted_tag in zip(batch_elt_expected, batch_elt_pred):
                    # Skip <?word_sep?> and <?pad?> tags, if any
                    if expected_tag.item() == WORD_SEP_IX or expected_tag.item() == SEQ_PAD_IX:
                        continue

                    predicted_this_batch.append(valid.ix_to_tag[predicted_tag.item()])
                    expected_this_batch.append(valid.ix_to_tag[expected_tag.item()])

                if valid.is_surface:
                    predicted_this_batch, expected_this_batch = align_seqs(predicted_this_batch, expected_this_batch,
                                                                           pad="PADDED")

                predicted.extend(predicted_this_batch)
                expected.extend(expected_this_batch)

        f1_micro = f1_score(expected, predicted, average="micro")
        f1_macro = f1_score(expected, predicted, average="macro")
        f1_weighted = f1_score(expected, predicted, average="weighted")

        return valid_loss, len(valid_loader), classification_report(expected, predicted,
                                                                    zero_division=0.0), f1_micro, f1_macro, f1_weighted


def _collate_by_padding(batch):
    # Check if the morphemes are divided into submorphemes
    # If so, we need to pad every item in the batch to the same length
    if batch[0][0].dim() == 2:
        longest_submorphemes = max(morphemes.size(dim=1) for morphemes, _tags in batch)
        for (i, (morphemes, tags)) in enumerate(batch):
            pad = nn.ConstantPad1d((0, longest_submorphemes - morphemes.size(dim=1)), SEQ_PAD_IX)
            batch[i] = pad(morphemes), tags

    morphemes = pad_sequence([morphemes for morphemes, tags in batch], batch_first=True, padding_value=SEQ_PAD_IX)
    expected_tags = pad_sequence([tags for morphemes, tags in batch], batch_first=True, padding_value=SEQ_PAD_IX)
    return (morphemes, expected_tags)


def train_model(model, name: str, config, train_set: AnnotatedCorpusDataset,
                valid: AnnotatedCorpusDataset, best_ever_macro_f1: float = 0.0, use_ray=True):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_set.to(device)
    valid.to(device)

    train_loader = DataLoader(
        train_set,
        batch_sampler=BatchSampler(RandomSampler(train_set), config["batch_size"], False),
        collate_fn=_collate_by_padding,
    )

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    start_epoch = 0
    if use_ray:
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
                model.load_state_dict(checkpoint_state["model_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    best_macro = 0.0
    best_macro_epoch = 0
    micro_at_best_macro = 0.0

    batches = len(train_loader)
    for epoch in range(start_epoch, config["epochs"]):
        # Set model to training mode (affects layers such as BatchNorm)
        model.train()

        train_loss = 0
        start = time.time()
        for morphemes, expected_tags in iter(train_loader):
            # Clear gradients
            model.zero_grad()

            # print(list(train_set.ix_to_tag[tag.item()] for tag in torch.flatten(expected_tags)))
            # print(list(train_set.ix_to_morpheme[morph.item()] for morph in torch.flatten(morphemes)))
            # print()

            # Calculate loss and backprop
            loss = model.loss(morphemes, expected_tags)
            train_loss += loss.item() * morphemes.size(dim=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()

        elapsed = time.time() - start
        print(f"Eval (elapsed = {elapsed:.2f}s)")
        elapsed = time.time() - start
        valid_loss, valid_batches, _report, f1_micro, f1_macro, f1_weighted = analyse_model(model, config, valid)
        print(f"Epoch {epoch} done in {elapsed:.2f}s. "
              f"Train loss: {train_loss / batches:.3f}. "
              f"Valid loss: {valid_loss / valid_batches:.3f}. "
              f"Micro F1: {f1_micro:.3f}. Macro f1: {f1_macro:.3f}")

        if f1_macro > best_macro:
            best_macro = f1_macro
            best_macro_epoch = epoch
            micro_at_best_macro = f1_micro

            out_dir = os.environ.get("MODEL_OUT_DIR")
            if out_dir and not use_ray and best_macro >= best_ever_macro_f1:
                print("Saving model")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, name) + ".pt", "wb") as f:
                    torch.save(EncapsulatedModel(name, model, train_set), f)

        if use_ray:
            checkpoint_data = {
                "epoch": epoch,
                "best_epoch": best_macro_epoch,
                "best_macro": best_macro,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"loss": valid_loss / valid_batches, "f1_macro": f1_macro, "f1_micro": f1_micro},
                    checkpoint=checkpoint,
                )

    _valid_loss, _valid_batches, report, f1_micro, f1_macro, f1_weighted = analyse_model(model, config, valid)
    print(f"{name}: Micro F1: {f1_micro}. Macro f1: {f1_macro}. Weighted F1: {f1_weighted}")
    print(f"Best Macro f1: {best_macro} in epoch {best_macro_epoch} (micro here was {micro_at_best_macro})")
    print(report)

    return f1_micro, f1_macro, f1_weighted


def tune_model(model, main_config, feature_level, name: str, epochs, trainset: AnnotatedCorpusDataset,
               valid: AnnotatedCorpusDataset, cpus=4, hrs=11):
    ray.init(num_cpus=cpus)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    embed_config, mk_embed = feature_level[1], feature_level[3]
    name, mk_model = model[0], model[1]

    config = {**main_config, **embed_config}

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        grace_period=3,
        reduction_factor=2,
    )

    trainset, valid = ray.put(trainset), ray.put(valid)
    result = tune.run(
        lambda conf: train_model(model_for_config(mk_model, mk_embed, ray.get(trainset), conf), name, conf,
                                 ray.get(trainset),
                                 ray.get(valid)),
        resources_per_trial={"gpu": 1.0 / cpus} if torch.cuda.is_available() else None,
        config=config,
        num_samples=100,
        time_budget_s=hrs * 60 * 60,  # 11h
        search_alg=BasicVariantGenerator(constant_grid_search=True, max_concurrent=4),
        scheduler=scheduler,
        storage_path=os.environ["TUNING_CHECKPOINT_DIR"],
    )

    for metric in ["f1_macro", "f1_micro"]:
        best_trial = result.get_best_trial(metric, "max", "all")
        print(f"Best trial by {metric}:")
        print(f" config: {best_trial.config}")
        print(f" val loss: {best_trial.last_result['loss']}")
        print(f" macro f1 {best_trial.last_result['f1_macro']}")
        print(f" micro {best_trial.last_result['f1_micro']}")

        best_model = model_for_config(mk_model, mk_embed, ray.get(trainset), best_trial.config)
        best_model.to(device)

        best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric=metric, mode="max")
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                best_checkpoint_data = pickle.load(fp)

            best_model.load_state_dict(best_checkpoint_data["model_state_dict"])
            _, _, report, f1_micro, f1_macro, f1_weighted = analyse_model(best_model, best_trial.config, ray.get(valid))
            print(f" {name}: Micro F1: {f1_micro}. Macro f1: {f1_macro}. Weighted F1: {f1_weighted}")
            print(
                f" {name}: Best macro f1: {best_checkpoint_data['best_macro']} at epoch {best_checkpoint_data['best_epoch']}")
            print(report)


def model_for_config(mk_model, mk_embed, trainset, config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    embed_module = mk_embed(config, trainset).to(device)
    model = mk_model(trainset, embed_module, config).to(device)
    return model
