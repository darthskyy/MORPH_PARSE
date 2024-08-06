import torch
import torch.nn as nn
import torch.nn.functional as function
from common import AnnotatedCorpusDataset, classes_only_no_tags, WORD_SEP_IX, SEQ_PAD_IX
from lstm import BiLSTMTagger


def tag_only(tag):
    digits = [(i, c) for i, c in enumerate(tag) if c.isdigit()]
    if len(digits) > 0:
        return tag[:digits[0][0]]
    else:
        return tag


def class_only(tag):
    digits = [(i, c) for i, c in enumerate(tag) if c.isdigit()]
    if len(digits) > 0:
        return tag[digits[0][0]:]
    else:
        return "NON_CLASS"


# TODO refactor!
class DummyTrainset:
    def __init__(self, num_tags):
        self.num_tags = num_tags


class BiLSTMNaiveTwoStepTagger(nn.Module):
    def __init__(self, embed, config, trainset: AnnotatedCorpusDataset):
        super(BiLSTMNaiveTwoStepTagger, self).__init__()
        self.full_tag_to_ix = trainset.tag_to_ix
        self.full_ix_to_tag = trainset.ix_to_tag

        self.class_to_ix = dict()
        self.class_ix_to_class = dict()

        self.tag_to_ix = dict()
        self.tag_ix_to_tag = dict()

        self.full_ix_to_class_only = dict()
        self.full_ix_to_tag_only = dict()

        for (full_ix, tag) in trainset.ix_to_tag.items():
            cl = class_only(tag)
            t = tag_only(tag)

            if full_ix not in self.full_ix_to_class_only:
                class_ix = self.class_to_ix.setdefault(cl, len(self.class_to_ix))
                self.full_ix_to_class_only[full_ix] = class_ix
                self.class_ix_to_class[class_ix] = cl

                tag_ix = self.tag_to_ix.setdefault(t, len(self.tag_to_ix))
                self.full_ix_to_tag_only[full_ix] = tag_ix
                self.tag_ix_to_tag[tag_ix] = t

        self.tag_tagger = BiLSTMTagger(embed, config, DummyTrainset(len(self.tag_ix_to_tag)))
        self.class_tagger = BiLSTMTagger(embed, config, DummyTrainset(len(self.class_ix_to_class)))

        self.map_tags_to_tag = torch.vmap(lambda i: self.full_ix_to_tag_only[i])
        self.map_classes_to_class = torch.vmap(lambda i: self.full_ix_to_class_only[i])

        self.loss_fn = nn.NLLLoss(ignore_index=SEQ_PAD_IX)

    def loss(self, morphemes, expected):
        expected_tags = expected.clone().to('cpu')
        expected_classes = expected.clone().to('cpu')

        expected_tags.apply_(lambda i: self.full_ix_to_tag_only[i])
        expected_classes.apply_(lambda i: self.full_ix_to_class_only[i])

        print(expected_tags)
        print(expected_classes)

        tags = self.tag_tagger(morphemes)
        classes = self.class_tagger(morphemes)

        print(len(self.full_ix_to_class_only))
        print(len(self.class_ix_to_class))
        print(len(self.tag_ix_to_tag))

        print(tags.size(), classes.size())

        # scores = self.forward(morphemes).transpose(1, 2)
        # return self.loss_fn(scores, expected)

    def forward_tags_only(self, xs):
        tags = self.tag_tagger.forward_tags_only(xs)
        classes = self.class_tagger.forward_tags_only(xs)

        preds = []
        # Split by batch
        for tags, classes in zip(torch.unbind(tags), torch.unbind(classes)):
            pred = []
            # This loop splits by morpheme
            for tag, clss in zip(tags, classes):
                tag_text = self.tag_ix_to_tag[tag.item()]
                class_text = self.class_ix_to_class[clss.item()]

                full = tag_text + class_text if class_text != "NON_CLASS" else tag_text
                # print(full)

                if full in self.full_tag_to_ix:
                    # print("In full tag to ix")
                    pred.append(self.full_tag_to_ix[full])
                elif tag_text in self.full_tag_to_ix:
                    # print("Tag text in full tag to ix")
                    pred.append(self.full_tag_to_ix[tag_text])
                else:
                    # print("not in any!!!!")
                    pred.append(WORD_SEP_IX)

            pred = torch.tensor(pred)
            preds.append(pred)

        return torch.stack(preds)

    def forward(self, morphemes):

        embeds = self.submorpheme_embeddings(morphemes)
        embeds = self.drop(embeds)

        lstm_out, hidden = self.lstm(embeds)
        lstm_out = self.drop(lstm_out)

        tag_space = self.hidden2tag(lstm_out)
        tag_scores = function.log_softmax(tag_space, dim=2)
        return tag_scores
