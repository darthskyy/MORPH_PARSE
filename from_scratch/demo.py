import torch

from common import AnnotatedCorpusDataset, split_words, split_sentences, tokenize_into_morphemes, \
    tokenize_into_chars, analyse_model, split_sentences_embedded_sep, _split_sentences_raw, WORD_SEP_TEXT, UNK_IDX

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

splits = {
    "word": split_words,
    "sentence": split_sentences,
    "sentence-embed": split_sentences_embedded_sep
}

tokenizations = {
    "morpheme": tokenize_into_morphemes,
    "char": tokenize_into_chars,
}

while True:
    # split = splits[input("Split [word / sentence / sentence-embed] > ")]
    # tokenize = tokenizations[input("Tokenize [morpheme / char] > ")]
    # lang = input("Language [XH / NR / ZU / SS] > ")
    # model_path = input("Model [path from out_models] > ")

    split = splits["sentence-embed"]
    tokenize = tokenizations["morpheme"]
    lang = "ZU"
    model_path = "bilstm-ZU-demo.pt"

    _, test = AnnotatedCorpusDataset.load_data(
        lang,
        split=split,
        tokenize=tokenize,
        use_testset=False,
    )

    model = torch.load("out_models/" + model_path, map_location=device)
    _, _, report, f1_micro, f1_macro, _ = analyse_model(
        model,
        {"batch_size": 1},
        test
    )
    print(report)
    print(f"Micro F1: {f1_micro}. Macro f1: {f1_macro}")

    while True:
        query = input("Morpological segmentation (separated by -) or Q to quit >")
        # query = "nga-tulu kwa-loko , ku-b-a khona ku-niket-el-a"
        if query.lower().strip() == "q":
            break

        words = query.replace(" ", "-" + WORD_SEP_TEXT + "-").split("-")
        all_encoded = []

        for morphemes, tags in split((words, ["???" for word in words])):
            for morpheme in morphemes:
                morpheme_encoded = []
                for submorpheme in tokenize(morpheme):
                    morpheme_encoded.append(test.morpheme_to_ix[submorpheme] if submorpheme in test.morpheme_to_ix else UNK_IDX)
                all_encoded.append(torch.tensor(morpheme_encoded))

            encoded = torch.stack([torch.stack(all_encoded, dim=0)], dim=0)

            print(
                [test.ix_to_tag[tag.item()] for tag in torch.flatten(model.forward_tags_only(encoded))]
            )
