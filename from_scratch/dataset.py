import re
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Constants for some foundational dataset elements (padding, word separators, unknown tokens)
SEQ_PAD_TEXT = "<?pad?>"
SEQ_PAD_IX = 0
WORD_SEP_IX = 1
WORD_SEP_TEXT = "<?word_sep?>"
UNK_IDX = 2


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


def split_sentences_raw(words):
    """Split raw text into sentences"""

    morphemes_acc, tags_acc = [], []
    for morphemes, tags in words:
        if len(morphemes_acc) != 0:
            morphemes_acc.append(WORD_SEP_TEXT)
            tags_acc.append(WORD_SEP_TEXT)

        morphemes_acc.extend(morphemes)
        tags_acc.extend(tags)

        if morphemes in [["."], ["!"], ["?"]] or [morphemes] in [["."], ["!"], ["?"]]:
            yield (morphemes_acc, tags_acc)
            morphemes_acc, tags_acc = [], []

    if len(morphemes_acc) != 0:
        yield (morphemes_acc, tags_acc)


def tokenize_into_morphemes(morpheme):
    """Tokenise a morpheme into its morphemes... which is just the morpheme itself"""
    return [morpheme]  # The word passed in is already a list of morphemes


def tokenize_into_lower_morphemes(morpheme):
    """Tokenise a word into its morphemes, lowercased (just the morpheme itself lowercased)"""
    return [morpheme.lower()]


def tokenize_into_chars(morpheme):
    """Tokenise a morphemes into its characters"""
    return list(morpheme) if morpheme != WORD_SEP_TEXT else [WORD_SEP_TEXT]


def tokenize_into_trigrams_with_sentinels(morpheme):
    """
    Tokenize a morpheme into its trigrams, using sentinels for the beginning and end.
    For instance, "ndi" would be tokenized into <START_MORPHEME, n, d>, <n, d, i>, <d, i, END_MORPHEME>
    """
    morpheme = ["START_MORPHEME"] + list(morpheme) + ["END_MORPHEME"] if morpheme != WORD_SEP_TEXT else [WORD_SEP_TEXT]
    return [(morpheme[i], morpheme[i + 1], morpheme[i + 2]) for i in range(len(morpheme) - 2)]


def tokenize_into_trigrams_no_sentinels(word):
    """
    Tokenize a morpheme into its trigrams, without using sentinels for the beginning and end.
    For instance, "enza" would be tokenized into <e, n, z>, <n, z, a>
    """
    word = list(word) if word != WORD_SEP_TEXT else [WORD_SEP_TEXT]
    if len(word) == 1:
        return [word[0], SEQ_PAD_TEXT, SEQ_PAD_TEXT]
    elif len(word) == 2:
        return [word[0], word[1], SEQ_PAD_TEXT]
    else:
        return [(word[i], word[i + 1], word[i + 2]) for i in range(len(word) - 2)]


def identity(x):
    """Simple identity function (used to map tags when wanting to use the full tagset)"""
    return x


def tags_only_no_classes(tag):
    """Map an NCHLT dataset grammatical tag to just the syntactic tag without noun classes. E.g., Dem14 -> Dem"""
    digits = [(i, c) for i, c in enumerate(tag) if c.isdigit()]
    if len(digits) > 0:
        return tag[:digits[0][0]]
    else:
        return tag


def classes_only_no_tags(tag):
    """Map an NCHLT dataset grammatical tag to just the noun class without syntactic tag. E.g., Dem14 -> 14"""
    digits = [(i, c) for i, c in enumerate(tag) if c.isdigit()]
    if len(digits) > 0:
        return tag[digits[0][0]:]
    else:
        return "NON_CLASS"


def prepare_sequence(seq, to_ix, device=None) -> torch.tensor:
    """Encode a given sequence as a tensor of indices (from the to_ix dict)"""
    idxs = [to_ix[w] if w in to_ix else UNK_IDX for w in seq]
    if device:
        return torch.tensor(idxs).to(device)
    else:
        return torch.tensor(idxs)


_DOUBLE_LABEL_PAT = re.compile("Pos[0-9]")


def _clean_double_labelled_morphemes(morpheme_seq: list, tag_seq: list):
    """Select the first tag of a morpheme in cases where they are doubly-tagged"""
    pos_tag_ix = [i for i, tag in enumerate(tag_seq) if _DOUBLE_LABEL_PAT.match(tag)]
    tag_seq.pop(pos_tag_ix[0])

    if len(morpheme_seq) != len(tag_seq):
        _clean_double_labelled_morphemes(morpheme_seq, tag_seq)


def extract_morphemes_and_tags_from_file_2022(filename: str, use_surface, is_demo=False):
    """Extract the morpheme & tag sequences from Gaustad & Puttkammer's 2022 dataset,
    'Linguistically annotated dataset for four official South African languages with a conjunctive orthography:
    IsiNdebele, isiXhosa, isiZulu, and Siswati' https://www.data-in-brief.com/article/S2352-3409(22)00205-0/fulltext
    """
    with open(filename) as f:
        for line in f.readlines():
            cols = line.strip().split("\t")
            morpheme_seq = cols[2].split("_")
            tag_seq = cols[3].split("_")

            if not use_surface:
                # Clean the double-labelled morphemes (by taking the 1st one) if this isn't a surface
                # segmentation. The surface segmentation comes pre-cleaned (see scripts/prep_surface.py)
                if not is_demo and len(morpheme_seq) != len(tag_seq):
                    _clean_double_labelled_morphemes(morpheme_seq, tag_seq)

                if not is_demo and len(morpheme_seq) != len(tag_seq):
                    print("Wrong len!", morpheme_seq, tag_seq)

            yield (morpheme_seq, tag_seq)


def inspect(x):
    """Debugging function which prints the value and then returns it."""
    print(x)
    return x


class AnnotatedCorpusDataset(Dataset):
    """
    `AnnotatedCorpusDataset` represents a loaded and parsed annotated corpus dataset (e.g. Gaustad & Puttkammer 2022).
    It contains all morpheme and tag sequences as well as dictionaries mapping from submorphemes and tags to indices.
    One instance of `AnnotatedCorpusDataset` will either be the training or validation/testing portion.
    """

    def __init__(self, seqs, num_submorphemes: int, num_tags: int, ix_to_tag, tag_to_ix, ix_to_morpheme, morpheme_to_ix,
                 is_surface, tokenize, split, lang):
        super().__init__()
        self.seqs = seqs
        self.num_submorphemes = num_submorphemes
        self.num_tags = num_tags
        self.ix_to_tag = ix_to_tag
        self.tag_to_ix = tag_to_ix
        self.ix_to_morpheme = ix_to_morpheme
        self.morpheme_to_ix = morpheme_to_ix
        self.is_surface = is_surface
        self.tokenize = tokenize
        self.split = split
        self.lang = lang

    @staticmethod
    def load_data(lang: str, use_surface=False, use_testset=False, split=split_words, tokenize=tokenize_into_morphemes,
                  map_tag=identity):
        """
        Load the data from the annotated corpus dataset, and return the training and validation portions.

        The major steps are as follows:
        1. Load raw train data
        2. Split into train/valid, OR load test data
        3. Count submorpheme frequencies (this refers to tokenization level - either morpheme or character)
        4. Create submorpheme<-->index and tag<-->index mappings
        5. Replace raw (text) data with indices and transform into tensors
        6. Return train & valid portions of dataset
        """

        # Initialise some index dictionaries used to map submorphemes & tags to indices
        submorpheme_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: WORD_SEP_IX,
                             "<?unk?>": UNK_IDX}  # unk accounts for unseen morphemes
        ix_to_submorpheme = {SEQ_PAD_IX: SEQ_PAD_TEXT, WORD_SEP_IX: WORD_SEP_TEXT, UNK_IDX: "<?unk?>"}
        tag_to_ix = {SEQ_PAD_TEXT: SEQ_PAD_IX, WORD_SEP_TEXT: WORD_SEP_IX}
        ix_to_tag = {SEQ_PAD_IX: SEQ_PAD_TEXT, WORD_SEP_IX: WORD_SEP_TEXT}
        submorpheme_frequencies = dict()

        training_data = []
        testing_data = []

        def insert_tags_into_dicts(tag_sequence, is_train):
            """Insert all tags from the given tag sequence into the tag <-> index dictionaries"""
            for tag in tag_sequence:
                tag = map_tag(tag)
                if tag not in tag_to_ix:
                    if not is_train:
                        print(f"Tag {tag} not found in trainset!")

                    ix = len(tag_to_ix)
                    tag_to_ix[tag] = ix
                    ix_to_tag[ix] = tag

        suffix = "_SURFACE" if use_surface else ""
        test_suffix = "SET_GOLD_SURFACE" if use_surface else ""

        # STEP 1: load dataset
        raw = extract_morphemes_and_tags_from_file_2022(f"data/TRAIN/{lang}_TRAIN{suffix}.tsv", use_surface)

        # We split by sentences in order to get a fair train/test split. We do have sentence level models, so we
        # want to ensure that we don't split words across sentences incorrectly.
        sentences = list(split_sentences_raw(raw))

        # STEP 2: split into validation / load test data
        # If not using the testset itself, automatically split of 10% of the data into the validation set. Otherwise,
        # load the testset from the file.
        if not use_testset:
            test_amount = len(sentences) // 10
            random.shuffle(sentences)
            test_sentences = sentences[:test_amount]
            train_sentences = sentences[test_amount:]
        else:
            print("Using testset")
            train_sentences = sentences
            test_sentences = list(
                split_sentences_raw(
                    extract_morphemes_and_tags_from_file_2022(f"data/TEST/{lang}_TEST{test_suffix}.tsv", use_surface)))

        # STEP 3: Count submorpheme frequencies.
        # We want to replace any submorpheme only seen once with `UNK` to improve generalization, so we only include
        # submorphemes which occur more than once
        for sentence in train_sentences:
            for (morphemes, tags) in split(sentence):
                for morpheme in morphemes:
                    for submorpheme in tokenize(morpheme):
                        submorpheme_frequencies.setdefault(submorpheme, 0)
                        submorpheme_frequencies[submorpheme] += 1

        # STEP 4: Create tag<-->index and submorpheme<-->index mappings
        for sentence in train_sentences:
            for (morphemes, tags) in split(sentence):
                # Insert submorphemes of morphemes from train set into the embedding indices
                # Replace those with only 1 occurence with UNK though
                for morpheme, tag in zip(morphemes, tags):
                    for submorpheme in tokenize(morpheme):
                        if submorpheme_frequencies[submorpheme] > 1:
                            submorpheme_to_ix.setdefault(submorpheme, len(submorpheme_to_ix))
                            ix_to_submorpheme.setdefault(len(submorpheme_to_ix) - 1, submorpheme)

                # Also insert tags into embedding indices
                insert_tags_into_dicts(tags, True)

                training_data.append((morphemes, tags))

        unseen_morpheme = set()
        for sentence in test_sentences:
            for (morphemes, tags) in split(sentence):
                # We skip inserting morphemes from the test set into the embedding indices, because it is realistic
                # that there may be unseen morphemes in the final data, of course

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

        def encode_dataset(dataset):
            """Encode the dataset into tensors by converting all submorphemes/tags to indices and padding them"""
            return [
                (pad_sequence([prepare_sequence(tokenize(m), submorpheme_to_ix, device) for m in morpheme_seq],
                              padding_value=SEQ_PAD_IX, batch_first=True),
                 prepare_sequence((map_tag(tag) for tag in tag_seq), tag_to_ix, device))
                for (morpheme_seq, tag_seq) in dataset
            ]

        # STEP 5: Transform to tensors
        training_data, testing_data = encode_dataset(training_data), encode_dataset(testing_data)

        print("train, test len:", len(training_data), len(testing_data))

        # STEP 6: done - just return one part as training and one as testing
        return (
            AnnotatedCorpusDataset(training_data, len(submorpheme_to_ix), len(tag_to_ix), ix_to_tag, tag_to_ix,
                                   ix_to_submorpheme, submorpheme_to_ix, use_surface, tokenize, split,
                                   lang),
            AnnotatedCorpusDataset(testing_data, len(submorpheme_to_ix), len(tag_to_ix), ix_to_tag, tag_to_ix,
                                   ix_to_submorpheme, submorpheme_to_ix, use_surface, tokenize, split,
                                   lang),
        )

    def to(self, device):
        """Move the dataset onto a specific device"""
        self.seqs = [(a.to(device), b.to(device)) for a, b in self.seqs]

    def __getitem__(self, item):
        return self.seqs[item]

    def __len__(self):
        return len(self.seqs)
