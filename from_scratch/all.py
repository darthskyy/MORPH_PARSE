import torch

from lstm import LSTMTagger
from common import AnnotatedCorpusDataset, train_model, split_words, split_sentences, tokenize_into_morphemes, tokenize_into_chars

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

for (split, split_name, epochs) in [(split_words, "word", 10), (split_sentences, "sentence", 40)]:
    for (tokenization, tokenization_name) in [(tokenize_into_morphemes, "morpheme"), (tokenize_into_chars, "character")]:
        for lang in ["XH", "NR", "SS", "ZU"]:
            train, test = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=tokenization)

            print(f"Training {split_name}-level, {tokenization_name}-tokenization bi-LSTM for {lang}")
            model = LSTMTagger(6, 6, train).to(device)

            train_model(model, f"lang-{lang}_split-{split_name}_tokenize-{tokenization_name}_model-bilstm", epochs, train, test)
