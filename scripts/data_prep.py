import re
import os

dirs = ['TEST', 'TRAIN']
out_name_format = "{0}_{1}.tsv"


def read_lines(file_path: str, skip_first=False) -> list:
    """Reads the lines of a file and returns them as a list."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return lines if not skip_first else lines[1:]

def format_line(line: str) -> str:
    """
    Removes the Lemma and POS columns from the line then adds the canonical segmentation of the word without the morphological tags.
    """
    line = line.rstrip()
    raw, parsed = line.split()[:2]
    segmentation, tags = split_tags(parsed)
    line = [raw, parsed, '_'.join(segmentation), '_'.join(tags)]
    return '\t'.join(line)


def split_tags(text: str) -> (str, list[str]):
    """Split a word into its canonical segmentation and morpheme tags."""
    split = [morpheme for morpheme in re.split(r'\[[a-zA-Z-_0-9|]*?]-?', text) if morpheme != ""]
    return (split, re.findall(r'\[([a-zA-Z-_0-9|]*?)]', text))


def write_lines(file_path: str, lines: list) -> None:
    """Writes a list of lines to a file."""
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    # assumes that the directory structure is as follows:
    # data/
    #   TEST/
    #     SADII.{lang}.*
    #   TRAIN/
    #     SADII.{lang}.*
    # script/
    #   data_prep.py

    # outputs the files in the following format:
    # word{tab}canonical_segmentation{tab}morphological_parse
    for dir in dirs:
        for in_file in os.listdir(f'../data/{dir}'):
            # Skip the English files
            if 'EN' in in_file or not in_file.endswith('.txt'):
                continue

            # Get the language of the file
            lang = in_file.split('.')[1]

            # Read the lines of the file and remove the lines with the <LINE#> tag
            lines = read_lines(os.path.join(f'../data/{dir}', in_file))
            lines = [line for line in lines if '<LINE#' not in line]
            lines = [format_line(line) for line in lines]

            # Write the formatted lines to a new file
            out_file = out_name_format.format(lang, dir)
            write_lines(os.path.join(f'../data/{dir}', out_file), lines)


if __name__ == '__main__':
    main()
