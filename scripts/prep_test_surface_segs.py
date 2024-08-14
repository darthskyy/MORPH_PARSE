import re
from data_prep import read_lines, split_tags, write_lines


def extract_from_gold(line: str) -> list[str]:
    split = line.split("\t")
    return [split[0], split[3]]


def prepare_test_lines(gold_lines: list[str], predicted_lines: list[str]) -> list[str]:
    out = []
    for gold, pred in zip(gold_lines, predicted_lines):
        gold, pred = gold.strip(), pred.strip()

        word, gold_tags = extract_from_gold(gold)
        pred_seg = pred.split("|")[1].strip()
        pred_seg = re.sub(r"\(.*\)", "", pred_seg).replace("-", "_")

        pred_seg = pred_seg or "-"  # Due to a limitation of the segmenter, "-" => "", so just fix it here

        out.append("\t".join([word, "_", pred_seg, gold_tags]))

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

    for lang, lang_id in langs.items():
        gold_path = f"data/TEST/{lang_id}_TEST.tsv"
        pred_path = f"data/Surface/predicted/{lang}.test.surface.predictions.conll"

        gold = read_lines(gold_path)
        pred = [line for line in read_lines(pred_path, skip_first=True) if line.strip()]
        write_lines(f"data/TEST/{lang_id}_TEST_SURFACE.tsv", prepare_test_lines(gold, pred))


if __name__ == "__main__":
    main()
