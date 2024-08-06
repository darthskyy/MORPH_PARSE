"""Much of this module is from Francois Meyer's pos-nguni bi-lstm crf tagger, used with permission

https://github.com/francois-meyer/pos-nguni/blob/main/src/bilistm_crf_tagger.py"""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from common import AnnotatedCorpusDataset, SEQ_PAD_IX
from from_scratch.bilstm_crf import CRF

class CrfTagger(nn.Module):
    def __init__(self, embed, config, trainset: AnnotatedCorpusDataset,
                 num_rnn_layers=1, rnn="lstm"):
        super(CrfTagger, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        self.vocab_size = trainset.num_submorphemes
        self.tagset_size = trainset.num_tags
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.embedding = embed
        self.drop = nn.Dropout(config["dropout"])
        self.crf = CRF(self.hidden_dim, self.tagset_size).to(self.dev)

    def __build_features(self, morphemes):
        masks = morphemes.any(dim=2)  # SEQ_PAD_IX is 0, so this checks which words have non-padding submorpheme IXs
        embeds = self.embedding(morphemes.long())
        embeds = self.drop(embeds)

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]
        embeds = self.drop(embeds)

        return embeds, masks

    def loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward_tags_only(self, xs):
        return self.forward(xs)[1]

    def forward(self, xs):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
