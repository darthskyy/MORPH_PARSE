import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .dataset import AnnotatedCorpusDataset, WORD_SEP_TEXT, UNK_IDX, SEQ_PAD_IX, WORD_SEP_IX


class EncapsulatedModel(nn.Module):
    """
    An EncapsulatedModel wraps a BiLSTMTagger or BiLSTMCrfTagger to accept segmented morphemes, thus performing
    any necessary data prep / mapping.
    """

    def __init__(
            self,
            name,
            model: nn.Module,
            dataset: AnnotatedCorpusDataset,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super(EncapsulatedModel, self).__init__()
        self.name = name
        self.lang = dataset.lang
        self.is_surface = dataset.is_surface
        self.ix_to_tag = dataset.ix_to_tag
        self.submorpheme_to_ix = dataset.morpheme_to_ix
        self.model = model
        self.tokenize = dataset.tokenize
        self.split = dataset.split
        self.device = device

    def to(self, device):
        super(EncapsulatedModel, self).to(device)
        self.model.to(device)
        self.device = device

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["device"]
        return state

    def __setstate__(self, state):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return super(EncapsulatedModel, self).__setstate__(state)

    def forward(self, segmented_sentences):
        """
        Morphologically analyse the given sentences. The input format is expected to be a `list[list[list[str]]]`. Each
        element of the list is a sentence, which is a list of words, which is a list of its morphemes.

        Returns a `list[list[list[str]]]` (same format as above but the str is replaced by the tag)
        """

        batches = []
        for segmented_sentence in segmented_sentences:
            # Convert the list-of-lists format to a separator format
            sentence = []
            for word in segmented_sentence:
                sentence.extend(word)
                sentence.append(WORD_SEP_TEXT)

            batches.append(sentence[:-1])  # Discard the last separator

        # Encode all sentences and batch together
        batch_sentences_encoded = []
        batches_morphemes = []
        for sentence in batches:
            for morphemes, _tags in self.split((sentence, [None for _ in range(0, len(sentence))])):
                sentence_encoded = []
                for morpheme in morphemes:
                    morpheme_encoded = []
                    for submorpheme in self.tokenize(morpheme):
                        morpheme_encoded.append(self.submorpheme_to_ix[submorpheme] if submorpheme in self.submorpheme_to_ix else UNK_IDX)
                    sentence_encoded.append(torch.tensor(morpheme_encoded, device=self.device))

                batches_morphemes.append(morphemes)
                batch_sentences_encoded.append(pad_sequence(sentence_encoded, padding_value=SEQ_PAD_IX, batch_first=True))

        # Run inference
        batches_encoded = pad_sequence(batch_sentences_encoded, padding_value=SEQ_PAD_IX, batch_first=True)
        batches_tagged = self.model.forward_tags_only(batches_encoded)

        tags_per_batch = []
        # Decode all sentences and re-batch together
        for sentence, morphemes in zip(torch.unbind(batches_tagged), batches_morphemes):
            sentence_tags = []
            word_tags = []
            for tag_ix, morpheme_text in zip(sentence.tolist(), morphemes):
                if morpheme_text == WORD_SEP_TEXT:
                    sentence_tags.append(word_tags)
                    word_tags = []
                    continue
                word_tags.append(self.ix_to_tag[tag_ix])

            if word_tags:
                sentence_tags.append(word_tags)

            tags_per_batch.append(sentence_tags)

        return tags_per_batch
