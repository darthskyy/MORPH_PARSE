from lstm import LSTMTagger
from common import AnnotatedCorpusDataset, train_model

for lang in ["XH", "NR", "SS", "ZU"]:
    for (split, split_name, epochs) in [(AnnotatedCorpusDataset.split_words, "word", 20), (AnnotatedCorpusDataset.split_sentences, "sentence", 50)]:
        train, test = AnnotatedCorpusDataset.load_data(lang, split)

        print(f"Training {split_name}-level bi-LSTM for {lang}")
        model = LSTMTagger(6, 6, train)
        train_model(model, f"{lang}_{split_name}_bi_lstm", epochs, train, test)
