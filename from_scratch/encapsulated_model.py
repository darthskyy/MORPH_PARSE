import torch
from torch import nn

from dataset import AnnotatedCorpusDataset, WORD_SEP_TEXT, UNK_IDX


class EncapsulatedModel(nn.Module):
    """
    An EncapsulatedModel wraps a BiLSTMTagger or BiLSTMCrfTagger to accept segmented morphemes, thus performing
    any necessary data prep / mapping.
    """

    def __init__(self, name, model: nn.Module, dataset: AnnotatedCorpusDataset):
        super(EncapsulatedModel, self).__init__()
        self.name = name
        self.lang = dataset.lang
        self.is_surface = dataset.is_surface
        self.ix_to_tag = dataset.ix_to_tag
        self.submorpheme_to_ix = dataset.morpheme_to_ix
        self.model = model
        self.tokenize = dataset.tokenize
        self.split = dataset.split

    def forward(self, segmented_sentence):
        """
        Morphologically analyse a given sentence. The input format is expected to be a `list[list[str]]`. Each element
        of the list is a word, and each word is a list of its morphemes.

        Returns a `list[list[str]]` where each element of the list is a word, and each word is a list of its morphemes'
        tags.
        """

        # Convert the list-of-lists format to a separator format
        sentence = []
        for word in segmented_sentence:
            sentence.extend(word)
            sentence.append(WORD_SEP_TEXT)

        sentence = sentence[:-1]  # Discard the last separator

        tags = []

        for morphemes, _tags in self.split((sentence, [None for _ in range(0, len(sentence))])):
            all_encoded = []
            for morpheme in morphemes:
                morpheme_encoded = []
                for submorpheme in self.tokenize(morpheme):
                    morpheme_encoded.append(self.submorpheme_to_ix[submorpheme] if submorpheme in self.submorpheme_to_ix else UNK_IDX)
                all_encoded.append(torch.tensor(morpheme_encoded))

            encoded = torch.stack([torch.stack(all_encoded, dim=0)], dim=0)

            tags.append([self.ix_to_tag[tag.item()] for tag in torch.flatten(self.model.forward_tags_only(encoded))])

        return tags[:-1]  # Discard the last separator
