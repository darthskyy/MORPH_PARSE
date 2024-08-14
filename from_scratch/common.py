import itertools
import os
import pickle
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
import random

import torch
from torch import optim, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from sklearn.metrics import f1_score, classification_report
from ray.train import Checkpoint, get_checkpoint
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator
from ray.util.client import ray

from aligned_f1 import align_seqs

SEQ_PAD_TEXT = "<?pad?>"
SEQ_PAD_IX = 0
WORD_SEP_IX = 1
WORD_SEP_TEXT = "<?word_sep?>"
UNK_IDX = 2

torch.manual_seed(0)
random.seed(0)


def split_words(sentence):
    """Split the corpus into words"""
    morphemes, tags = sentence[0], sentence[1]
    morphemes_acc, tags_acc = [], []
    for morpheme, tag in zip(morphemes, tags):
        if morpheme == WORD_SEP_TEXT:
            yield (morphemes_acc, tags_acc)
            morphemes_acc, tags_acc = [], []
        else:
            morphemes_acc.append(morpheme)
            tags_acc.append(tag)

    yield (morphemes_acc, tags_acc)

def split_sentences(sentence):
    """Split the corpus into sentences"""
    yield sentence


def split_sentences_embedded_sep(sentence):
    """Split the corpus into sentences, separating by adding '^' to the morpheme that begins each word"""
    morphemes, tags = sentence[0], sentence[1]

    morphemes_new = []
    tags_new = []

    is_start = True
    for morpheme, tag in zip(morphemes, tags):
        if morpheme == WORD_SEP_TEXT:
            is_start = True
            continue

        if is_start:
            morphemes_new.append("^" + morpheme)
            is_start = False
        else:
            morphemes_new.append(morpheme)

        tags_new.append(tag)

    yield morphemes_new, tags_new


def split_sentences_no_sep(sentence):
    """Split the corpus into sentences without word separators"""
    morphemes, tags = sentence[0], sentence[1]
    morphemes_new = [morpheme for morpheme, tag in zip(morphemes, tags) if morpheme != WORD_SEP_TEXT]
    tags_new = [tag for morpheme, tag in zip(morphemes, tags) if morpheme != WORD_SEP_TEXT]
    yield morphemes_new, tags_new


def _split_sentences_raw(words):
    """Split raw text into sentences"""

    morphemes_acc, tags_acc = [], []
    for morphemes, tags in words:
        if len(morphemes_acc) != 0:
            morphemes_acc.append(WORD_SEP_TEXT)
            tags_acc.append(WORD_SEP_TEXT)

        morphemes_acc.extend(morphemes)
        tags_acc.extend(tags)

        if morphemes in [["."], ["!"], ["?"]]:
            yield (morphemes_acc, tags_acc)
            morphemes_acc, tags_acc = [], []

    if len(morphemes_acc) != 0:
        yield (morphemes_acc, tags_acc)


def tokenize_into_morphemes(word):
    return [word]


def tokenize_into_lower_morphemes(word):
    return [word.lower()]


def tokenize_into_chars(word):
    return list(word) if word != WORD_SEP_TEXT else [WORD_SEP_TEXT]


def tokenize_into_trigrams_with_sentinels(word):
    word = ["START_MORPHEME"] + list(word) + ["END_MORPHEME"] if word != WORD_SEP_TEXT else [WORD_SEP_TEXT]
    return [(word[i], word[i + 1], word[i + 2]) for i in range(len(word) - 2)]


def tokenize_into_trigrams_no_sentinels(word):
    word = list(word) if word != WORD_SEP_TEXT else [WORD_SEP_TEXT]
    if len(word) == 1:
        return [word[0], SEQ_PAD_TEXT, SEQ_PAD_TEXT]
    elif len(word) == 2:
        return [word[0], word[1], SEQ_PAD_TEXT]
    else:
        return [(word[i], word[i + 1], word[i + 2]) for i in range(len(word) - 2)]


def identity(x):
    return x


def tags_only_no_classes(tag):
    digits = [(i, c) for i, c in enumerate(tag) if c.isdigit()]
    if len(digits) > 0:
        return tag[:digits[0][0]]
    else:
        return tag


def classes_only_no_tags(tag):
    digits = [(i, c) for i, c in enumerate(tag) if c.isdigit()]
    if len(digits) > 0:
        return tag[digits[0][0]:]
    else:
        return "NON_CLASS"


class AnnotatedCorpusDataset(Dataset):
    def __init__(self, seqs, num_submorphemes: int, num_tags: int, ix_to_tag, tag_to_ix, ix_to_morpheme, morpheme_to_ix, tag_weights, is_surface):
        super().__init__()
        self.seqs = seqs
        self.num_submorphemes = num_submorphemes
        self.num_tags = num_tags
        self.ix_to_tag = ix_to_tag
        self.tag_to_ix = tag_to_ix
        self.ix_to_morpheme = ix_to_morpheme
        self.morpheme_to_ix = morpheme_to_ix
        self.tag_weights = tag_weights
        self.is_surface = is_surface

    @staticmethod
    def load_data(lang: str, use_surface=False, use_testset=False, split=split_words, tokenize=tokenize_into_morphemes, map_tag=identity,
                  use_2024=False, supp_training_langs=None):
        if supp_training_langs is None:
            supp_training_langs = []

        submorpheme_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: WORD_SEP_IX,
                             "<?unk?>": UNK_IDX}  # unk accounts for unseen morphemes
        ix_to_submorpheme = {SEQ_PAD_IX: SEQ_PAD_TEXT, WORD_SEP_IX: WORD_SEP_TEXT, UNK_IDX: "<?unk?>"}
        tag_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: WORD_SEP_IX}
        ix_to_tag = {SEQ_PAD_IX: SEQ_PAD_TEXT, WORD_SEP_IX: WORD_SEP_TEXT}
        submorpheme_frequencies = dict()
        tag_frequencies = dict()

        training_data = []
        testing_data = []

        double_label_pat = re.compile("Pos[0-9]")

        def clean_double_labelled_morphemes(morpheme_seq: list, tag_seq: list):
            pos_tag_ix = [i for i, tag in enumerate(tag_seq) if double_label_pat.match(tag)]
            tag_seq.pop(pos_tag_ix[0])

            if len(morpheme_seq) != len(tag_seq):
                clean_double_labelled_morphemes(morpheme_seq, tag_seq)

        def insert_tags_into_dicts(tag_sequence, is_train):
            for tag in tag_sequence:
                tag = map_tag(tag)
                if tag not in tag_to_ix:
                    if not is_train:
                        print(f"Tag {tag} not found in trainset!")

                    ix = len(tag_to_ix)
                    tag_to_ix[tag] = ix
                    ix_to_tag[ix] = tag

        def extract_morphemes_and_tags_from_file_2022(filename: str):
            with open(filename) as f:
                for line in f.readlines():
                    cols = line.strip().split("\t")
                    morpheme_seq = cols[2].split("_")
                    tag_seq = cols[3].split("_")

                    if not use_surface:
                        # Clean the double-labelled morphemes (by taking the 1st one) if this isn't a surface
                        # segmentation. The surface segmentation comes pre-cleaned (see scripts/prep_surface.py)
                        if len(morpheme_seq) != len(tag_seq):
                            clean_double_labelled_morphemes(morpheme_seq, tag_seq)

                        if len(morpheme_seq) != len(tag_seq):
                            print("Wrong len!", morpheme_seq, tag_seq)

                    yield (morpheme_seq, tag_seq)

        def split_tags(text: str):
            """Split a word into its canonical segmentation and morpheme tags."""
            canon = [morph for morph in re.split(r'\[[a-zA-Z-_0-9]*?]-?', text) if morph != ""]
            return (canon, re.findall(r'\[([a-zA-Z-_0-9]*?)]', text))

        def extract_morphemes_and_tags_from_file_2024(filename: str):
            with open(filename) as f:
                for line in f.readlines():
                    if line.startswith("<LINE"):
                        continue

                    analysis = line.strip().split()[1]
                    morpheme_seq, tag_seq = split_tags(analysis)

                    if len(morpheme_seq) != len(tag_seq):
                        clean_double_labelled_morphemes(morpheme_seq, tag_seq)

                    if len(morpheme_seq) != len(tag_seq):
                        print("Wrong len!", morpheme_seq, tag_seq)
                    yield (morpheme_seq, tag_seq)

        # First, we split by sentences in order to get a fair train/test split
        suffix = "_SURFACE" if use_surface else ""
        raw = extract_morphemes_and_tags_from_file_2022(f"data/TRAIN/{lang}_TRAIN{suffix}.tsv")
        if use_2024:
            raw = itertools.chain(
                raw,
                extract_morphemes_and_tags_from_file_2024(f"data/NEW/{lang}_NEW.txt")
            )

        sentences = list(_split_sentences_raw(raw))

        if not use_testset:
            # Split the data in two
            test_amount = len(sentences) // 10
            random.shuffle(sentences)
            test_sentences = sentences[:test_amount]
            train_sentences = sentences[test_amount:]
        else:
            print("Using testset")
            train_sentences = sentences
            test_sentences = list(
                _split_sentences_raw(extract_morphemes_and_tags_from_file_2022(f"data/TEST/{lang}_TEST{suffix}.tsv")))

        supp = []
        for supp_lang in supp_training_langs:
            raw_supp = itertools.chain(
                extract_morphemes_and_tags_from_file_2024(f"data/NEW/{supp_lang}_NEW.txt"),
                extract_morphemes_and_tags_from_file_2022(f"data/TRAIN/{supp_lang}_TRAIN{suffix}.tsv")
            )
            supp = list(_split_sentences_raw(raw_supp))

        # Count submorpheme & tag frequencies
        for sentence in train_sentences:
            for (morphemes, tags) in split(sentence):
                for morpheme in morphemes:
                    for submorpheme in tokenize(morpheme):
                        submorpheme_frequencies.setdefault(submorpheme, 0)
                        submorpheme_frequencies[submorpheme] += 1
                for i, tag in enumerate(tags):
                    tag = map_tag(tag)
                    tag_frequencies.setdefault(tag, 0)
                    tag_frequencies[tag] += 1

        for sentence in supp:
            for (morphemes, tags) in split(sentence):
                for morpheme in morphemes:
                    for submorpheme in tokenize(morpheme):
                        submorpheme_frequencies.setdefault(submorpheme, 0)
                        submorpheme_frequencies[submorpheme] += 1

        for sentence in supp:
            for (morphemes, tags) in split(sentence):
                # Insert submorphemes of morphemes from train set into the embedding indices
                # Replace those with only 1 occurence with UNK though
                for morpheme in morphemes:
                    for submorpheme in tokenize(morpheme):
                        if submorpheme_frequencies[submorpheme] > 1:
                            submorpheme_to_ix.setdefault(submorpheme, len(submorpheme_to_ix))
                            ix_to_submorpheme.setdefault(len(submorpheme_to_ix) - 1, submorpheme)

        for sentence in train_sentences:
            for (morphemes, tags) in split(sentence):
                # Insert submorphemes of morphemes from train set into the embedding indices
                # Replace those with only 1 occurence with UNK though
                for morpheme, tag in zip(morphemes, tags):
                    for submorpheme in tokenize(morpheme):
                        # TODO try and lowercase before too
                        non_grammar_tags = ["Foreign", "Num", "NStem", "VRoot", "AuxVStem", "VerbTerm", "Abbr", "ProperName", "AdjStem", "Intrans", "RelStem", "Ideoph"]
                        if submorpheme_frequencies[submorpheme] > 1:
                            submorpheme_to_ix.setdefault(submorpheme, len(submorpheme_to_ix))
                            ix_to_submorpheme.setdefault(len(submorpheme_to_ix) - 1, submorpheme)
                        # elif tag in non_grammar_tags:
                        #     print("'", submorpheme, f"' (tag {tag}) only appeared once! replacing with UNK", sep="")

                # Also insert tags into embedding indices
                insert_tags_into_dicts(tags, True)

                training_data.append((morphemes, tags))

        unseen_morpheme = set()
        for sentence in test_sentences:
            for (morphemes, tags) in split(sentence):
                # We skip inserting morphemes from the test set into the embedding indices, because it is realistic
                # that there may be unseen morphemes

                # However, we _do_ insert tags, since we know all the tags upfront. Technically we should just have
                # a predefined list, but that's annoying to do
                insert_tags_into_dicts(tags, False)

                testing_data.append((morphemes, tags))

                for morpheme in morphemes:
                    for submorpheme in tokenize(morpheme):
                        if submorpheme not in submorpheme_to_ix:
                            unseen_morpheme.add(submorpheme)

        print(f"{(len(unseen_morpheme) / len(submorpheme_to_ix)) * 100.0}% submorphemes not found in train!")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Encode a given sequence as a tensor of indices (from the to_ix dict)
        def prepare_sequence(seq, to_ix) -> torch.tensor:
            idxs = [to_ix[w] if w in to_ix else UNK_IDX for w in seq]
            return torch.tensor(idxs).to(device)

        def encode_dataset(dataset):
            return [
                (pad_sequence([prepare_sequence(tokenize(m), submorpheme_to_ix) for m in morpheme_seq],
                              padding_value=SEQ_PAD_IX, batch_first=True),
                 prepare_sequence((map_tag(tag) for tag in tag_seq), tag_to_ix))
                for (morpheme_seq, tag_seq) in dataset
            ]

        # Encode everything
        training_data, testing_data = encode_dataset(training_data), encode_dataset(testing_data)

        print("train, test len:", len(training_data), len(testing_data))

        lowest_freq = sorted(freq for freq in tag_frequencies.values())[0]
        tag_frequencies = [(tag_to_ix[tag], freq) for tag, freq in tag_frequencies.items()]
        tag_frequencies = [freq for tag, freq in sorted(tag_frequencies, key=lambda t: t[0])]

        for ix in tag_to_ix.values():
            if ix >= len(tag_frequencies):
                tag_frequencies.append(lowest_freq)

        tag_frequencies = [1.0 / float(freq + 1) for freq in tag_frequencies]
        tag_frequencies = torch.tensor(tag_frequencies)

        return (
            AnnotatedCorpusDataset(training_data, len(submorpheme_to_ix), len(tag_to_ix), ix_to_tag, tag_to_ix,
                                   ix_to_submorpheme, submorpheme_to_ix, tag_frequencies, use_surface),
            AnnotatedCorpusDataset(testing_data, len(submorpheme_to_ix), len(tag_to_ix), ix_to_tag, tag_to_ix,
                                   ix_to_submorpheme, submorpheme_to_ix, tag_frequencies, use_surface),
        )

    def to(self, device):
        self.seqs = [(a.to(device), b.to(device)) for a, b in self.seqs]

    def __getitem__(self, item):
        return self.seqs[item]

    def __len__(self):
        return len(self.seqs)


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
                    predicted_this_batch, expected_this_batch = align_seqs(predicted_this_batch, expected_this_batch, pad="PADDED")

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

        if f1_macro > best_macro :
            best_macro = f1_macro
            best_macro_epoch = epoch
            micro_at_best_macro = f1_micro

            out_dir = os.environ.get("MODEL_OUT_DIR")
            if out_dir and not use_ray and best_macro >= best_ever_macro_f1:
                print("Saving model")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, name) + ".pt", "wb") as f:
                    torch.save(model, f)

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
