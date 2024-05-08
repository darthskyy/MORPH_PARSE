import torch
import torch.nn as nn
import torch.nn.functional as function
from common import AnnotatedCorpusDataset


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, trainset: AnnotatedCorpusDataset, combine_submorphemes=None):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.combine_submorphemes = combine_submorphemes
        self.submorpheme_embeddings = nn.Embedding(trainset.num_submorphemes, embedding_dim)

        # The LSTM takes morpheme embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        # self.feedforward = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.hidden2tag = nn.Linear(hidden_dim * 2, trainset.num_tags)
        self.hidden_state = None
        self.init_hidden_state()

    def init_hidden_state(self):
        num_directions = 2
        h = torch.zeros(num_directions, 1, self.hidden_dim, requires_grad=True)
        c = torch.zeros(num_directions, 1, self.hidden_dim, requires_grad=True)
        self.hidden_state = (h, c)

    def forward(self, morphemes):
        # print("morphemes", morphemes)
        # print("first morpheme", morphemes[0])
        # print("embeddings for first", self.submorpheme_embeddings(morphemes[0]).view(len(morphemes[0]), 1, -1))

        embeds = torch.stack([torch.sum(self.submorpheme_embeddings(subwords).view(len(subwords), 1, -1), dim=0) for subwords in morphemes], dim=0)
        lstm_out, hidden = self.lstm(embeds, self.hidden_state)

        # stuff = self.feedforward(lstm_out.view(len(sentence), -1))
        tag_space = self.hidden2tag(lstm_out.view(len(embeds), -1))
        tag_scores = function.log_softmax(tag_space, dim=1)
        return tag_scores
