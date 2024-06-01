import argparse
import re
import os

dirs = ['TEST', 'TRAIN']
in_name_format = 'SADII.{0}.Morph_Lemma_POS.1.0.0.{1}.CTexT.TG.2021-09-30'
out_name_format = "{0}_{1}.tsv"

def read_lines(file_path: str) -> list:
    """Reads the lines of a file and returns them as a list."""
    with open(file_path, 'r') as f:
        return f.readlines()

def format_line(line: str) -> str:
    """
    Removes the Lemma and POS columns from the line then adds the canonical segmentation of the word without the morphological tags.
    """
    line = line.rstrip()
    line = line.split('\t')[:2]
    line.append(line[1])
    line[1] = strip_tags(line[1])

    return '\t'.join(line)

def strip_tags(text: str) -> str:
    """Removes morphological tags from a parsed word."""
    return re.sub(r'\[.*?\]', '', text)

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

    parser = argparse.ArgumentParser(description='Format the SADII files.')
    parser.add_argument('--data', type=str, default='../data', help='The directory containing the SADII files.')
    parser.add_argument('--clean', action='store_true', help='Remove the source files after formatting them.')
    args = parser.parse_args()

    if args.data == '../data':
        print("Please specify the directory containing the SADII files.")
        return

    # outputs the files in the following format:
    # word{tab}canonical_segmentation{tab}morpholgical_parse
    for dir in dirs:
        for in_file in os.listdir(os.path.join(args.data, dir)):
            # Skip the English files
            if 'EN' in in_file:
                continue

            # Get the language of the file
            lang = in_file.split('.')[1]

            # Read the lines of the file and remove the lines with the <LINE#> tag
            lines = read_lines(os.path.join(os.path.join(args.data, dir), in_file))
            lines = [line for line in lines if '<LINE#' not in line]
            lines = [format_line(line) for line in lines]
            lines = ["word\tseg\tparse"] + lines

            # Write the formatted lines to a new file
            out_file = out_name_format.format(lang, dir)
            write_lines(os.path.join(args.data, dir, out_file), lines)

            # Remove the source file if the --clean flag is set
            if args.clean:
                os.remove(os.path.join(os.path.join(args.data, dir), in_file))

    if args.clean:
        # remove the Protocols dire
        print("Source files removed.")

if __name__ == '__main__':
    main()