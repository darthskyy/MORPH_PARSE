import time
import pprint as pp
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from typing import TypeAlias, Any
from torcheval.metrics.functional import multiclass_f1_score

TextTaggedSequenced: TypeAlias = tuple[list[str], list[str]]

WORD_SEP_IX = 0
WORD_SEP_TEXT = "<?word_sep?>"


def split_words(words):
    for morphemes, tags in words:
        yield (morphemes, tags)


def split_sentences(words):
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


def _resplit_sentences(sentences):
    for morphemes, tags in sentences:
        morphemes_acc, tags_acc = [], []
        for morpheme, tag in zip(morphemes, tags):
            if morpheme == WORD_SEP_TEXT:
                yield (morphemes_acc, tags_acc)
                morphemes_acc, tags_acc = [], []
            else:
                morphemes_acc.append(morpheme)
                tags_acc.append(tag)


def tokenize_into_morphemes(word):
    return [word]


def tokenize_into_chars(word):
    return list(word) if word != WORD_SEP_TEXT else [WORD_SEP_TEXT]


def combine_singleton_tensor(submorpheme_embeddings: torch.tensor):
    assert submorpheme_embeddings.size() == (1,)
    return submorpheme_embeddings.item()


def combine_by_summing(submorpheme_embeddings: torch.tensor):
    return torch.sum(submorpheme_embeddings)


class AnnotatedCorpusDataset(Dataset):
    def __init__(self, seqs: list[tuple[int, int]], num_submorphemes: int, num_tags: int, ix_to_tag: dict[int, str]):
        super().__init__()
        self.seqs = seqs
        self.num_submorphemes = num_submorphemes
        self.num_tags = num_tags
        self.ix_to_tag = ix_to_tag

    @staticmethod
    def load_data(lang: str, split=split_words, tokenize=tokenize_into_morphemes):
        submorpheme_to_ix = {WORD_SEP_TEXT: 0, "<?unk?>": 1}  # unk accounts for unseen morphemes
        tag_to_ix = {WORD_SEP_TEXT: WORD_SEP_IX}
        ix_to_tag = {WORD_SEP_IX: WORD_SEP_TEXT}

        training_data = []
        testing_data = []

        def insert_tags_into_dicts(tag_sequence):
            for tag in tag_sequence:
                if tag not in tag_to_ix:
                    ix = len(tag_to_ix)
                    tag_to_ix[tag] = ix
                    ix_to_tag[ix] = tag

        def extract_morphemes_and_tags_from_file(filename: str):
            with open(filename) as f:
                for line in f.readlines():
                    cols = line.strip().split("\t")
                    morpheme_seq = cols[2].split("_")
                    tag_seq = cols[3].split("_")

                    if len(morpheme_seq) != len(tag_seq):
                        continue  # TODO why are there any like this in the first place?

                    yield (morpheme_seq, tag_seq)

        # First, we split by sentences in order to get a fair train/test split
        sentences = list(split_sentences(extract_morphemes_and_tags_from_file(f"data/TRAIN/{lang}_TRAIN.tsv")))

        # Split the data in half
        test_amount = len(sentences) // 10
        test_sentences = _resplit_sentences(sentences[:test_amount])
        train_sentences = _resplit_sentences(sentences[test_amount:])

        for (morphemes, tags) in split(train_sentences):
            # Insert submorphemes of morphemes from train set into the embedding indices
            for morpheme in morphemes:
                for submorpheme in tokenize(morpheme):
                    submorpheme_to_ix.setdefault(submorpheme, len(submorpheme_to_ix))

            # Also insert tags into embedding indices
            insert_tags_into_dicts(tags)

            training_data.append((morphemes, tags))

        for (morphemes, tags) in split(test_sentences):
            # We skip inserting morphemes from the test set into the embedding indices, because it is realistic
            # that there may be unseen morphemes

            # However, we _do_ insert tags, since we know all the tags upfront. Technically we should just have
            # a predefined list, but that's annoying to do
            insert_tags_into_dicts(tags)

            testing_data.append((morphemes, tags))

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Encode a given sequence as a tensor of indices (from the to_ix dict)
        def prepare_sequence(seq: list[str], to_ix: dict[str, int]) -> torch.tensor:
            idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
            return torch.tensor(idxs).to(device)

        def encode_dataset(dataset: list[tuple[str, str]]) -> list[tuple[torch.tensor, torch.tensor]]:
            return [
                ([prepare_sequence(tokenize(m), submorpheme_to_ix) for m in morpheme_seq],
                 prepare_sequence(tag_seq, tag_to_ix))
                for (morpheme_seq, tag_seq) in dataset
            ]

        # Encode everything
        training_data, testing_data = encode_dataset(training_data), encode_dataset(testing_data)

        return (
            AnnotatedCorpusDataset(training_data, len(submorpheme_to_ix), len(tag_to_ix), ix_to_tag),
            AnnotatedCorpusDataset(testing_data, len(submorpheme_to_ix), len(tag_to_ix), ix_to_tag),
        )

    def __getitem__(self, item):
        return self.seqs[item]

    def __len__(self):
        return len(self.seqs)


def _analyse_model(model, test: AnnotatedCorpusDataset) -> tuple[float, float, float, list[tuple[str, Any]]]:
    with torch.no_grad():
        # Set model to evaluation mode (affects layers such as BatchNorm)
        model.eval()

        print("Evaluating model...")
        start = time.time()
        predicted = []
        expected = []

        for morphemes, expected_tags in test:
            tag_scores = model(morphemes)

            for expected_tag, morpheme in zip(expected_tags, tag_scores):
                # Skip <?word_sep?> tags, if any
                if expected_tag == WORD_SEP_IX:
                    continue

                predicted.append(torch.argmax(morpheme).item() - 1)
                expected.append(expected_tag - 1)

        print(f"Took {time.time() - start:.2f}s")

        predicted = torch.tensor(predicted, dtype=torch.long)
        expected = torch.tensor(expected, dtype=torch.long)

        # -1 to num tags to remove word_sep
        f1_scores = multiclass_f1_score(predicted, expected, num_classes=test.num_tags - 1, average=None)
        f1_scores = ((test.ix_to_tag[tag_ix + 1], score.item()) for tag_ix, score in enumerate(f1_scores))
        f1_micro = multiclass_f1_score(predicted, expected, num_classes=test.num_tags - 1, average="micro").item()
        f1_macro = multiclass_f1_score(predicted, expected, num_classes=test.num_tags - 1, average="macro").item()
        f1_weighted = multiclass_f1_score(predicted, expected, num_classes=test.num_tags - 1, average="weighted").item()

        return f1_micro, f1_macro, f1_weighted, sorted(f1_scores, key=lambda pair: -pair[1])


def train_model(model, name: str, epochs: int, train: AnnotatedCorpusDataset, test: AnnotatedCorpusDataset):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train = DataLoader(train, shuffle=True, batch_size=1, collate_fn=lambda x: x[0])

    for epoch in range(epochs):
        # Set model to training mode (affects layers such as BatchNorm)
        model.train()

        start = time.time()
        for morphemes, expected_tags in iter(train):
            # Clear gradients
            model.zero_grad()

            # Run model on this word's morphological segmentation
            scores = model(morphemes)

            # Calculate loss and backprop
            loss = loss_function(scores, expected_tags)
            loss.backward()
            optimizer.step()

            # Reset the model's hidden state
            model.init_hidden_state()

        elapsed = time.time() - start
        eta = elapsed * (epochs - epoch)
        print(f"Epoch {epoch} done in {elapsed:.2f}s. ETA: {eta:.2f}")

    f1_micro, f1_macro, f1_weighted, f1_all = _analyse_model(model, test)
    print(f"Micro F1: {f1_micro}. Macro f1: {f1_macro}. Weighted F1: {f1_weighted}. All: {pp.pformat(f1_all)}")
