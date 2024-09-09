class ValidsetLanguageResults:
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.seed_best_epochs = []

    def __repr__(self):
        return f"ValidsetLanguage(lang={self.lang_code}, best_epochs={self.seed_best_epochs})"


class Epoch:
    def __init__(self, epoch_number, macro, micro):
        self.epoch_number = epoch_number
        self.macro = macro
        self.micro = micro

    def __repr__(self):
        return f"Epoch(num={self.epoch_number}, macro={self.macro:.3f}, micro={self.micro:.3f})"


class TestsetLanguageResults:
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.seeds = []

    def __repr__(self):
        return f"TestsetLanguage(lang={self.lang_code}, seeds={self.seeds})"

class TestsetSeedResults:
    def __init__(self):
        self.epochs = []

    def __repr__(self):
        return f"Seed(epochs={self.epochs})"



def parse_validset_results(lines) -> dict[str, ValidsetLanguageResults]:
    langs = dict()
    current_lang = None

    for line in lines:
        if line.startswith("Training"):
            lang_code = line[-2:]
            current_lang = langs.setdefault(lang_code, ValidsetLanguageResults(lang_code))
        elif line.startswith("Best Macro f1:"):
            epoch = line.split(" ")
            epoch_number, macro, micro = int(epoch[6]), float(epoch[3]), float(epoch[-1][:-1])
            current_lang.seed_best_epochs.append(Epoch(epoch_number, macro, micro))
        else:
            continue

    return langs


def parse_testset_results(lines) -> dict[str, TestsetLanguageResults]:
    langs = dict()
    current_lang = None
    current_seed = None

    for line in lines:
        if line.startswith("Training"):
            lang_code = line[-2:]
            current_lang = langs.setdefault(lang_code, TestsetLanguageResults(lang_code))
            current_seed = TestsetSeedResults()
        elif line.startswith("Epoch "):
            epoch = line.split(" ")
            epoch_number, macro, micro = int(epoch[1]), float(epoch[-1]), float(epoch[-4][:-1])
            current_seed.epochs.append(Epoch(epoch_number, macro, micro))
        elif line.startswith("Best Macro f1"):
            current_lang.seeds.append(current_seed)
            current_seed = TestsetSeedResults()
        else:
            continue

    return langs


def lang_sort_key(lang):
    langs = ["ZU", "NR", "XH", "SS"]
    return langs.index(lang)


if __name__ == "__main__":
    path = "out_models/crf_sentence-embed_morpheme_canon"
    with open(f"{path}/out_crf_morpheme_sentence_embed-notest.txt") as valid_out:
        valid_lines = valid_out.read().splitlines(keepends=False)
        valid_res = parse_validset_results(valid_lines)

    with open(f"{path}/out_crf_morpheme_sentence_embed-test.txt") as test_out:
        test_lines = test_out.read().splitlines(keepends=False)
        test_res = parse_testset_results(test_lines)

    langs = dict()
    latex = []
    for (lang, lang_valid) in valid_res.items():
        lang_test = test_res[lang]

        macro, micro = 0, 0
        for seed_best_epoch, test_epochs in zip(lang_valid.seed_best_epochs, lang_test.seeds):
            test_epoch = test_epochs.epochs[seed_best_epoch.epoch_number]
            micro += test_epoch.micro
            macro += test_epoch.macro

        n_seeds = len(lang_valid.seed_best_epochs)
        macro, micro = macro / n_seeds, micro / n_seeds
        print(f"{lang} Macro F1 {macro:.3f}, Micro F1 {micro:.3f}")
        langs[lang] = (macro, micro)

    langs = list(langs.items())
    langs.sort(key=lambda item: lang_sort_key(item[0]))
    latex = [f"{macro:.3f} & {micro:.3f}" for (lang, (macro, micro)) in langs]

    print(" & ".join(latex))