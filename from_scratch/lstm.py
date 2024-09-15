import torch
import torch.nn as nn
import torch.nn.functional as function
from common import AnnotatedCorpusDataset, SEQ_PAD_IX


class BiLSTMTagger(nn.Module):
    """
    A `BiLSTMTagger` uses a bidirectional long short-term memory model to classify a given sequence of morphemes
    with their corresponding grammatical tags
    """

    def __init__(self, embed, config, trainset: AnnotatedCorpusDataset):
        super(BiLSTMTagger, self).__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.hidden_dim = config["hidden_dim"]
        self.submorpheme_embeddings = embed.to(self.dev)

        # The LSTM takes morpheme embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed.output_dim, self.hidden_dim, batch_first=True, bidirectional=True, device=self.dev)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, trainset.num_tags, device=self.dev)
        self.drop = nn.Dropout(config["dropout"])
        self.loss_fn = nn.NLLLoss(ignore_index=SEQ_PAD_IX)

    def loss(self, morphemes, expected):
        # Run model on this word's morphological segmentation
        scores = self.forward(morphemes).transpose(1, 2)
        return self.loss_fn(scores, expected)

    def forward_tags_only(self, xs):
        return torch.argmax(self.forward(xs), dim=2)

    def forward(self, morphemes):
        embeds = self.submorpheme_embeddings(morphemes)
        embeds = self.drop(embeds)

        lstm_out, hidden = self.lstm(embeds)
        lstm_out = self.drop(lstm_out)

        tag_space = self.hidden2tag(lstm_out)
        tag_scores = function.log_softmax(tag_space, dim=2)
        return tag_scores
