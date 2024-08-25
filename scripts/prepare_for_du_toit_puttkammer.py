for lang in ["ZU", "XH", "NR", "SS"]:
    with open(f"data/TRAIN/{lang}_TEST.tsv") as in_file:
        raw = [line.split("\t")[0] for line in in_file.read().splitlines() if line.strip()]

    with open(f"data/DuToitPuttkammer/{lang}_TEST.txt", "w") as out_file:
        out_file.write(" ".join(raw))
