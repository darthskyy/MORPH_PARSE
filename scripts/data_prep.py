"""
Authors:
    - Cael Marquard
    - Simbarashe Mawere
Date:
    - 2024/07/07

Description:
    - This script downloads, reads the SADII files and formats them into a TSV format.
    - The TSV format is as follows:
        - word{tab}canonical_segmentation{tab}morphological_parse
    - The script assumes that the directory structure is as follows:
        - data/
            - TEST/
                - SADII.{lang}.*
            - TRAIN/
                - SADII.{lang}.*
        - script/
            - data_prep.py
    - The script outputs the files in the following format:
        - word{tab}canonical_segmentation{tab}morphological_parse
    - The script skips the English files.
"""

import argparse
import re
import io
import os
import requests
import zipfile

# for extracting the data
DIRS = ["TEST", "TRAIN"]
OUT_NAME_FORMAT = "{0}_{1}.tsv"

# for downloading the data
URL = "https://repo.sadilar.org/bitstream/handle/20.500.12185/546/SADII_CTexT_2022-03-14.zip?sequence=4&isAllowed=y"
OUT_DIR = "data"

def download_and_extract(file_url: str, out_dir: str):
    """
    Downloads a zip file and extracts it into a specified directory

    Args:
        file_url (str): The URL of the zip file to download
        out_dir (str): The directory to extract the contents of the zip file into

    Returns:
        None
    """
    r = requests.get(file_url)
    content = io.BytesIO(r.content)

    with zipfile.ZipFile(content, "r") as zip_ref:
        zip_ref.extractall(out_dir)

def remove_extras(out_dir: str):
    """
    Removes unnecessary files and directories from the extracted zip file. Made for the SADII_CTexT dataset

    Args:
        out_dir (str): The directory containing the extracted files

    Returns:
        None
    """
    # removing unnecessary file
    os.system(f"rmdir {os.path.join(out_dir, "Protocols")} /s /q")
    os.remove(os.path.join(out_dir, "README.Data.txt"))

    for sub_dir in os.listdir(out_dir):
        for file_ in os.listdir(os.path.join(out_dir, sub_dir)):
            if "EN" in file_:
                os.remove(os.path.join(out_dir, sub_dir, file_))


def read_lines(file_path: str) -> list:
    """
    Reads the lines of a file and returns them as a list.

    Args:
        file_path (str): The path to the file to read

    Returns:
        list: The lines of the file
    """
    with open(file_path, "r") as f:
        return f.readlines()


def format_line(line: str) -> str:
    """
    Removes the Lemma and POS columns from the line then adds the canonical segmentation of the word without the morphological tags.

    Args:
        line (str): The line to format

    Returns:
        str: The formatted line

    Example:
        format_line("a[DET]b[N]") -> "a\tb\ta_b\tDET_N"
    """
    line = line.rstrip()
    raw, parsed = line.split()[:2]
    segmentation, tags = split_tags(parsed)
    line = [raw, parsed, "_".join(segmentation), "_".join(tags)]
    return "\t".join(line)


def split_tags(text: str) -> tuple[str, list[str]]:
    """
    Split a word into its canonical segmentation and morpheme tags.

    Args:
        text (str): The word to split

    Returns:
        (str, list[str]): The canonical segmentation and morpheme tags

    Example:
        split_tags("a[DET]b[N]") -> (["a", "b"], ["DET", "N"])
    """
    split = [
        morpheme
        for morpheme in re.split(r"\[[a-zA-Z-_0-9]*?]-?", text)
        if morpheme != ""
    ]
    return (split, re.findall(r"\[([a-zA-Z-_0-9]*?)]", text))


def write_lines(file_path: str, lines: list) -> None:
    """
    Writes a list of lines to a file.

    Args:
        file_path (str): The path to the file to write to
        lines (list): The lines to write to the file

    Returns:
        None
    """
    with open(file_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Download and extract a zip file")
    parser.add_argument("--url", type=str, default=URL, help="The URL of the zip file to download")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR, help="The directory to extract the contents of the zip file into")
    parser.add_argument("--clean", action="store_true", help="Remove the unnecessary files and directories")
    args = parser.parse_args()

    download_and_extract(args.url, args.out_dir)
    if args.clean:
        remove_extras(args.out_dir)
    print("Download and extraction complete!")

    for dir in DIRS:
        for in_file in os.listdir(os.path.join(args.out_dir, dir)):
            # Skip the English files
            if "EN" in in_file or not in_file.endswith(".txt"):
                continue

            # Get the language of the file
            lang = in_file.split(".")[1]

            # Read the lines of the file and remove the lines with the <LINE#> tag
            lines = read_lines(os.path.join(args.out_dir, dir, in_file))
            lines = [line for line in lines if "<LINE#" not in line]
            lines = [format_line(line) for line in lines]

            # Write the formatted lines to a new file
            out_file = OUT_NAME_FORMAT.format(lang, dir)
            write_lines(os.path.join(args.out_dir, dir, out_file), lines)
            if args.clean:
                os.remove(os.path.join(args.out_dir, dir, in_file))


if __name__ == "__main__":
    main()
