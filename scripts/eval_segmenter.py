from sklearn.metrics import f1_score
from aligned_f1 import align_seqs


def process_file(path):
    with open(path) as f:
        return [line.split("\t")[2].lower().split("_") for line in f.read().splitlines(keepends=False)]


for lang_id in ["ZU", "NR", "SS", "XH"]:
    # Because we _evaluate_ on the segmenter output, the _test_ set file actually has the _predicted_ output (for segmenting)
    # Bit odd at first, but if you think about it, it makes sense.
    pred_raw = process_file(f"data/TEST/{lang_id}_TEST_SURFACE.tsv")
    gold_raw = process_file(f"data/TEST/{lang_id}_TESTSET_GOLD_SURFACE.tsv")
    # pred_raw = process_file(f"data/TEST/{lang_id}_TEST_CANONICAL_PRED.tsv")
    # gold_raw = process_file(f"data/TEST/{lang_id}_TEST.tsv")

    pred_aligned, gold_aligned = [], []

    for pred_word, gold_word in zip(pred_raw, gold_raw):
        pred_word, gold_word = align_seqs(pred_word, gold_word)
        pred_aligned.extend(pred_word)
        gold_aligned.extend(gold_word)

    micro = f1_score(gold_aligned, pred_aligned, average="micro")
    macro = f1_score(gold_aligned, pred_aligned, average="macro")

    print(f"Lang: {lang_id}. Micro F1: {micro:.4f}. Macro F1: {macro:.4f}")
