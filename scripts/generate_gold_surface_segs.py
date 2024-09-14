# Adapted from Jan Buys's script for the MorphParse project. Presumably licensed as something GPLv2 compatible, since
# the Levenshtein package has that license. We will defer to whatever license Jan's version is under and license this
# file under that (for this file ONLY)

# Jan Buys
# Process Sadilar morphologically annotated isiZulu data

# regular expressions with help from ChatGPT
import re
import json
from collections import defaultdict

# pip install levenshtein
import Levenshtein


def get_surface_segmentation(surface_str, canonical_morphemes):
    canonical_str = ''.join(canonical_morphemes)  # without seperator
    c_str_to_morph_ind = []
    for i, morph in enumerate(canonical_morphemes):
        for _ in morph:
            c_str_to_morph_ind.append(i)

    edits = Levenshtein.opcodes(canonical_str.lower(), surface_str.lower())
    alignment = [-1 for _ in canonical_str]
    surface_alignment = [-1 for _ in surface_str]

    insert_ind = -1

    for e in edits:
        if e[0] == 'delete':
            for i in range(e[1], e[2]):
                alignment[i] = -1
        elif e[0] == 'equal' or e[0] == 'replace':
            if insert_ind != -1:
                insert_ind = -1  # don't do anything else here for now
            assert e[2] - e[1] == e[4] - e[3]
            for i in range(e[1], e[2]):
                alignment[i] = e[3] + (i - e[1])
            for i in range(e[3], e[4]):
                surface_alignment[i] = e[1] + (i - e[3])
        elif e[0] == 'insert':
            assert insert_ind == -1  # assume no consecutive inserts
            insert_ind = e[1]
            for i in range(e[3], e[4]):
                # align dashes to previous morpheme if possible, but other inserts to next morpheme
                if surface_str[e[3]] == '-' and surface_alignment[i - 1] >= 0:
                    surface_alignment[i] = surface_alignment[i - 1]
                elif e[1] == len(canonical_str):
                    surface_alignment[i] = e[1] - 1
                else:
                    surface_alignment[i] = e[1]

    surface_morpheme_alignment = list(map(lambda l: -1 if l == -1 else c_str_to_morph_ind[l], surface_alignment))
    surface_canonical_morphemes = ['' for _ in canonical_morphemes]
    surface_segmentation = []
    for i, a in enumerate(surface_morpheme_alignment):
        if i == 0:
            surface_segmentation.append('B')
        elif surface_str[i] == '-':  # dashes should always attach to previous morpheme
            surface_morpheme_alignment[i] = surface_morpheme_alignment[i - 1]
            surface_segmentation.append('I')
        else:
            if a == surface_morpheme_alignment[i - 1]:
                surface_segmentation.append('I')
            else:
                surface_segmentation.append('B')

        assert surface_morpheme_alignment[i] != -1
        surface_canonical_morphemes[surface_morpheme_alignment[i]] += surface_str[i]

    return surface_canonical_morphemes


class MorphWord:
    def __init__(self, line):
        entry = line.strip().split()
        assert len(entry) == 4
        self.surface, morph_str, self.lemma, pos = entry
        pos_list = list(re.findall(r'([A-Za-z]+)(\d*a*)', pos)[0])  # split off noun class
        self.pos, self.nclass = pos_list[0], pos_list[1]  # .replace("0", "") # note: not really using this nclass
        # split on - as long as following ]
        if self.pos == 'PUNC':
            self.punct = self.surface
        else:
            self.punct = ''
            morphs_lab = re.split(r'(?<=\])-', morph_str)
            # split morpheme and [label]
            morphs_lab_split = [re.split(r'(\[[^\[\]]*\])', ml) for ml in morphs_lab]
            morphs_lab_split = [[x for x in ml if x] for ml in morphs_lab_split]  # remove empty entries
            self.canonical_morphemes = [ml[0] for ml in morphs_lab_split]
            canonical_labels_list = []
            # split off noun class numbers
            # canonical_labels_list = [list(re.findall(r'^([A-Za-z-]+)(\d|\d\d|\da|\dps\dpp)$', ml[1])[0]) for ml in morphs_lab_split]
            for ml in morphs_lab_split:
                refind = re.findall(r'^\[([A-Za-z-]+)(\d?|\d\d|\da|\dps|\dpp)\]$', ml[1])
                if len(refind) > 0:
                    entry = list(refind[0])
                    if 'ps' in entry[1] or 'pp' in entry[1]:  # move to be a feature rather than noun class
                        entry.insert(1, '')
                    canonical_labels_list.append(entry)
                else:
                    print('no match:', ml[1])

            self.canonical_labels = [cl[0] + cl[1] for cl in canonical_labels_list]
            self.canonical_nclass = [cl[1] for cl in canonical_labels_list]
            self.canonical_xfeats = [cl[2] if len(cl) == 3 else "" for cl in canonical_labels_list]

            self.surface_canonical_morphemes = get_surface_segmentation(self.surface, self.canonical_morphemes)

    def new_tokens_ud_representation(self):
        new_tokens = []

        surface = ''
        lemma = ''
        upos = ''
        postag = ''
        feats = []
        can_morphs = []
        can_labels = []

        for i, can_morph in enumerate(self.canonical_morphemes):
            can_label = self.canonical_labels[i]

            if surface:
                new_tokens.append({
                    'form': surface,
                    'lemma': lemma,
                    'upos': '_',
                    'xpos': 'X' if postag is None else postag,
                    'feats': '_',
                    'misc': '-'.join([m + '[' + l + ']' for m, l in zip(can_morphs, can_labels)])
                })

            surface = self.surface_canonical_morphemes[i]
            can_morphs = [can_morph]
            can_labels = [can_label]
            lemma = can_morph
            postag = can_label

        new_tokens.append({
            'form': surface,
            'lemma': lemma,
            'upos': upos,
            'xpos': postag,
            'feats': '|'.join(set(feats)) if feats else '_',
            'misc': '-'.join([m + '[' + l + ']' for m, l in zip(can_morphs, can_labels)])
        })

        return new_tokens

    def morpheme_entry(self):
        morphemes = []
        for i, surf in enumerate(self.surface_canonical_morphemes):
            morphemes.append({'surface': surf,
                              'morph': self.canonical_morphemes[i],
                              'label': self.canonical_labels[i],
                              'nclass': self.canonical_nclass[i]})
        return morphemes


if __name__ == '__main__':
    base_path = 'data/'

    for tset in ["train", "test"]:
        for lang_code in ['ZU', 'NR', 'XH', 'SS']:
            if tset == "train":
                sadilar_train_fname = base_path + 'TRAIN/SADII.' + lang_code + '.Morph_Lemma_POS.1.0.0.TRAIN.CTexT.TG.2021-09-30.txt'
                out_conllu_fname = base_path + 'TRAIN/' + lang_code + '_TRAIN_SURFACE.tsv'
            else:
                sadilar_train_fname = base_path + 'TEST/SADII.' + lang_code + '.Morph_Lemma_POS.1.0.0.TEST.CTexT.TG.2021-09-30.txt'
                out_conllu_fname = base_path + 'TEST/' + lang_code + '_TESTSET_GOLD_SURFACE.tsv'

            write_outfiles = True

            sentences = []
            paragraph_id = ''
            sentence_surface = ''
            current_tokens = []
            sent_id = 1
            sentence_end = False

            with open(sadilar_train_fname, 'r') as file:
                for line in file:
                    if sentence_end:
                        sentences.append({'paragraph_id': paragraph_id, 'sentence_id': sent_id, 'text': sentence_surface,
                                          'tokens': current_tokens})
                        current_tokens = []
                        sentence_surface = ''
                        sentence_end = False
                        sent_id += 1
                    if line.startswith('<LINE#'):
                        paragraph_id = line.strip()[len('<LINE# '):-1]
                    else:
                        morphword = MorphWord(line)
                        morphemes = [] if morphword.punct else morphword.morpheme_entry()
                        if morphword.punct:
                            ud_rep = [{'form': morphword.surface, 'lemma': morphword.surface, 'upos': 'PUNCT', 'xpos': 'Punc',
                                       'feats': '', 'misc': f'{morphword.surface}[Punc]'}]
                            if morphword.surface in ('.', '!', '?'):
                                sentence_end = True
                            if morphword.surface in ('.', '!', '?', ':', ';', ','):  # no space before
                                sentence_surface += morphword.surface
                            elif morphword.surface in ('"', "'", '(', ')', '[', ']', '/', '-'):  # don't normalise for now
                                sentence_surface += ' ' + morphword.surface
                            else:
                                print(morphword.surface)
                                sentence_surface += ' ' + morphword.surface
                        else:
                            ud_rep = morphword.new_tokens_ud_representation()
                            if sentence_surface:
                                sentence_surface += ' ' + morphword.surface
                            else:
                                sentence_surface = morphword.surface

                        current_tokens.append({
                            'form': morphword.surface,
                            'pos': morphword.pos,
                            'punct': morphword.punct,
                            'morphemes': morphemes,
                            'ud': ud_rep
                        })

                if sentence_end or current_tokens:
                    sentences.append({'paragraph_id': paragraph_id, 'sentence_id': sent_id, 'text': sentence_surface,
                                      'tokens': current_tokens})
                    current_tokens = []
                    sentence_end = False

            if write_outfiles:
                with open(out_conllu_fname, 'w') as outfile:
                    for sent in sentences:
                        for token in sent['tokens']:
                            # Target output format: word {tab} surface_analysis {tab} surface_morphs_split {tag} surface_tags_split
                            word = token['form']

                            tags = [rep['xpos'] for rep in token['ud']]
                            morphs = [rep['form'] for rep in token['ud']]
                            analysis = "-".join([f"{morph}[{tag}]" for tag, morph in zip(tags, morphs)])

                            outfile.write('\t'.join([word, analysis, "_".join(morphs), "_".join(tags)]) + '\n')

