import csv
import re

from sklearn.metrics import classification_report, f1_score

from data_prep import split_tags
from scripts.evaluation import eval_model_aligned_multiset

with open("data/ZulMorph/ZulMorph_ZU_TEST.tsv") as f:
    zulmorph_lines = list(csv.reader(f, delimiter='\t', quotechar='"'))[1:-1]  # Skip header & closing LF

with open("data/TEST/ZU_TEST.tsv") as f:
    gold_lines = list(csv.reader(f, delimiter='\t', quotechar='"'))


def normalize_zm_tag(zm_tag: str, idx: int, context_morphemes: list[str]) -> list[str]:
    # - ComplExt? Rewrote to IntensExt
    zm_tag = zm_tag.replace("PC", "PossConc") \
        .replace("VTSubj", "VerbTerm") \
        .replace("VTNeg", "VerbTerm") \
        .replace("VT", "VerbTerm") \
        .replace("QC", "QuantConc") \
        .replace("RCPT", "RelConc") \
        .replace("RC", "RelConc") \
        .replace("AdjPre", "AdjPref") \
        .replace("AC", "AdjPref") \
        .replace("SCPT", "SC") \
        .replace("PotNeg", "NegPre") \
        .replace("EC", "EnumConc") \
        .replace("Item", "Foreign") \
        .replace("LongPres", "Pres") \
        .replace("Punct", "Punc") \
        .replace("SCHort", "SC") \
        .replace("SCSit", "SC") \
        .replace("ComplExt", "IntensExt")

    if zm_tag.startswith("NStem"):
        # NCHLT dataset doesn't have classes for noun stems
        return ["NStem"]
    elif zm_tag == "Hlon":
        # NCHLT dataset doesn't have Hlonipha as a tag
        return []
    elif zm_tag == "VerbTermPerf":
        # NCHLT dataset marks il[Perf]-e[VerbTerm] and e[VerbTerm],
        # whilst ZulMorph does ile[VTPerf] and e[VTPerf]

        # Only 3 cases: olusetshenziswe, isetshenziswe, engeke
        if idx >= len(context_morphemes):
            return ["VerbTerm"]

        return ["Perf", "VerbTerm"] if context_morphemes[idx] == "ile" else ["VerbTerm"]
    else:
        return [zm_tag]

def classification_scores():
    all_gold = []
    all_zm = []

    for (zulmorph_line, gold_line) in zip(zulmorph_lines, gold_lines):
        zm_raw, zm_analysis = zulmorph_line
        gold_raw, _, _, gold_tags = gold_line
        gold_tags = gold_tags.split("_")
        assert zm_raw == gold_raw

        # Concatenate adjacent noun class tags, e.g [BPre][10] -> [BPre10]
        zm_analysis = re.sub(r"]\[(?=[0-9])", "", zm_analysis)
        zm_seg, zm_tags = split_tags(zm_analysis)  # Things with \t+? (that ZulMorph can't segment) get zm_tags = []
        zm_tags = [final_tag for i, t in enumerate(zm_tags) for final_tag in normalize_zm_tag(t, i, zm_seg)]

        all_gold.append(gold_tags)
        all_zm.append(zm_tags)

    f1_micro, f1_macro = eval_model_aligned_multiset(all_zm, all_gold)
    print(f"Classification micro F1: {f1_micro:.4f}, macro f1: {f1_macro:.4f}")


def segmentation_scores():
    all_gold = []
    all_zm = []

    for (zulmorph_line, gold_line) in zip(zulmorph_lines, gold_lines):
        zm_raw, zm_analysis = zulmorph_line
        gold_raw, _, gold_seg, _ = gold_line
        gold_seg = gold_seg.split("_")
        assert zm_raw == gold_raw

        zm_seg, zm_tags = split_tags(zm_analysis)
        zm_seg = [zm_seg[0].replace("\t+?", "")] if len(zm_seg) == 1 else zm_seg

        if len(zm_tags) > 0 and (zm_tags[0] in ("Punct", "Num", "Item")):
            zm_seg = [zm_raw]

        all_gold.append(gold_seg)
        all_zm.append(zm_seg)

    f1_micro, f1_macro = eval_model_aligned_multiset(all_zm, all_gold)
    print(f"Segmentation micro F1: {f1_micro:.4f}, macro f1: {f1_macro:.4f}")


classification_scores()
segmentation_scores()
