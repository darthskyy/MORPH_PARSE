import os

from data_prep import read_lines, split_tags, write_lines


def prepare_line(line: str, is_test=False) -> list[str]:
    split = line.split(" | ", maxsplit=4)

    if len(split) == 1:
        # Probably whitespace-separated - some of the FOREIGN tags do this??
        split = line.split()
        return [split[0], split[3], split[0], split[3]]

    word, surface_seg, surface_with_tags, _ = split

    # Sometimes, morphs are double-tagged (e.g yo[SC4|Fut])
    # We can't train for multiple tags, but we can test for them
    tags = []
    for tag in split_tags(surface_with_tags)[1]:
        split = tag.split("|")
        if is_test:
            tags.extend(split)
        else:
            tags.append(split[0])

    return [word, surface_with_tags, surface_seg.replace("-", "_"), "_".join(tags)]


def prepare_train_lines(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        line = line.strip()
        if line:
            out.append("\t".join(prepare_line(line)))
    return out


def prepare_test_lines(gold_lines: list[str], predicted_lines: list[str]) -> list[str]:
    out = []
    for gold, pred in zip(gold_lines, predicted_lines):
        gold, pred = gold.strip(), pred.strip()
        if gold == "":
            assert pred == "", "Gold and predictions are mismatched!"
            continue

        word, surface_with_tags, _, gold_tags = prepare_line(gold, is_test=True)
        surface_seg = gold.split("|")[1]
        out.append("\t".join([word, surface_with_tags, surface_seg.replace("-", "_"), gold_tags]))

    return out


def main():
    # Assumes that the directory structure is as follows:
    # MORPH_PARSE/
    #   data/
    #     Surface/
    #       gold/
    #         {lang}.{test|train}.morphparse.conll
    #       predicted/
    #         {lang}.test.surface.predictions.conll
    #     TEST/
    #         ...
    # CWD should be MORPH_PARSE when invoking this script

    # outputs the files in the following format:
    # word{\t}morphological_analysis{\t}surface_segmentation{\t}morphological_tags

    langs = {
        "ndebele": "NR",
        "swati": "SS",
        "xhosa": "XH",
        "zulu": "ZU"
    }

    for lang in langs.keys():
        # Train set
        lines = read_lines(f"data/Surface/gold/{lang}.train.morphparse.conll", skip_first=True)
        prepped_lines = prepare_train_lines(lines)  # Prepare lines & skip empty
        write_lines(f"data/TRAIN/{langs[lang]}_TRAIN_SURFACE.tsv", prepped_lines)

        # Test set
        gold_test = read_lines(f"data/Surface/gold/{lang}.test.morphparse.conll", skip_first=True)
        predicted_test = read_lines(f"data/Surface/predicted/{lang}.test.surface.predictions.conll", skip_first=True)
        write_lines(f"data/TRAIN/{langs[lang]}_TEST_SURFACE.tsv", prepare_test_lines(gold_test, predicted_test))


if __name__ == "__main__":
    main()
