from common import AnnotatedCorpusDataset, SEQ_PAD_IX
import torch
from torch import nn
import torch.nn.functional as function

class BiLSTMCombinedTagger(nn.Module):
    def __init__(self, embed, config, trainset: AnnotatedCorpusDataset, tag_tagger, class_tagger):
        super(BiLSTMCombinedTagger, self).__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.hidden_dim = config["hidden_dim"]
        self.submorpheme_embeddings = embed.to(self.dev)

        self.tag_tagger = tag_tagger
        self.class_tagger = class_tagger

        for param in self.tag_tagger.parameters():
            param.requires_grad = False

        for param in self.class_tagger.parameters():
            param.requires_grad = False

        # The LSTM takes morpheme embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            embed.output_dim + tag_tagger.hidden2tag.out_features + class_tagger.hidden2tag.out_features,
            self.hidden_dim,
            batch_first=True,
            bidirectional=True,
            device=self.dev
        )

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

        tag_features = self.tag_tagger(morphemes)
        class_features = self.class_tagger(morphemes)

        # print(embeds.size(), tag_features.size(), class_features.size())
        in_features = torch.cat((embeds, tag_features, class_features), 2)

        lstm_out, hidden = self.lstm(in_features)
        lstm_out = self.drop(lstm_out)

        tag_space = self.hidden2tag(lstm_out)
        tag_scores = function.log_softmax(tag_space, dim=2)
        return tag_scores
