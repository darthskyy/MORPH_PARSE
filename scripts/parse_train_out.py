from typing import Tuple


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


def parse_validset_results(lines) -> Tuple[dict[str, ValidsetLanguageResults], list[str]]:
    langs = dict()
    current_lang = None
    config_lines = None

    for line in lines:
        if line.startswith("Config: "):
            config = line[len("Config: "):]

            # Bit of a hack to split { and } onto own lines
            config = config.replace("{", "{, ").replace("}", ", }")
            config = config.replace("'gradient_clip': inf", "'gradient_clip': float('inf')")
            config_lines = config.split(", ")
        elif line.startswith("Training"):
            lang_code = line[-2:]
            current_lang = langs.setdefault(lang_code, ValidsetLanguageResults(lang_code))
        elif line.startswith("Best Macro f1:"):
            epoch = line.split(" ")
            epoch_number, macro, micro = int(epoch[6]), float(epoch[3]), float(epoch[-1][:-1])
            current_lang.seed_best_epochs.append(Epoch(epoch_number, macro, micro))
        else:
            continue

    return langs, config_lines


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


def main():
    path = "out_models/old/bilstm_word_char_sum_canon"
    print(path)
    with open(f"{path}/out_bilstm_char_sum-notest.txt") as valid_out:
        valid_lines = valid_out.read().splitlines(keepends=False)
        valid_res, config_lines = parse_validset_results(valid_lines)

    langs = dict()
    for (lang, lang_valid) in valid_res.items():
        all_best_epochs = []
        for seed_best_epoch in lang_valid.seed_best_epochs:
            all_best_epochs.append(seed_best_epoch.epoch_number)
        mean_epoch = round(sum(all_best_epochs) / 5)
        langs[lang] = mean_epoch

    for line in config_lines:
        if "'epochs'" in line:
            print("    'epochs': {")
            for lang, mean in sorted(langs.items(), key=lambda lang_and_epoch: lang_and_epoch[0]):
                print(f'        "{lang}": {mean},')
            print("    },")
        elif line in ["{", "}"]:
            print(line)
        else:
            print(f"    {line},")


if __name__ == "__main__":
    main()
