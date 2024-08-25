from sklearn.metrics import f1_score
from aligned_f1 import align_seqs


for lang_id in ["ZU", "NR", "SS", "XH"]:
    def inspect(x):
        print(x)
        return x

    with open(f"data/TEST/{lang_id}_TEST_SURFACE.tsv") as f:
        lines = f.read().splitlines(keepends=False)
        pred_tags = [line.split("\t")[3].split("_") for line in lines]
        pred_segs = [line.split("\t")[2].split("_") for line in lines]
        pred_tags = [[tag for tag, _seg in zip(tag, seg)] for tag, seg in zip(pred_tags, pred_segs)]

    with open(f"data/TEST/{lang_id}_TEST.tsv") as f:
        gold_tags = [line.split("\t")[3].split("_") for line in f.read().splitlines(keepends=False)]

    pred_aligned, gold_aligned = [], []

    for pred_word, gold_word in zip(pred_tags, gold_tags):
        pred_word, gold_word = align_seqs(pred_word, gold_word)
        pred_aligned.extend(pred_word)
        gold_aligned.extend(gold_word)

    micro = f1_score(gold_aligned, pred_aligned, average="micro")
    macro = f1_score(gold_aligned, pred_aligned, average="macro")

    print(f"Lang: {lang_id}. Micro F1: {micro:.4f}. Macro F1: {macro:.4f}")
