import torch
from ray import tune

from bilstm_class_tagger import BiLSTMNaiveTwoStepTagger
from common import split_sentences, tokenize_into_morphemes, EmbedSingletonFeature, AnnotatedCorpusDataset, \
    train_model, model_for_config, analyse_model, classes_only_no_tags, tags_only_no_classes, \
    split_sentences_embedded_sep

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = (
    "bilstm",
    lambda train_set, embed, config: BiLSTMNaiveTwoStepTagger(embed, config, train_set)
)
splits = (split_sentences_embedded_sep, "sentences_embed", 20)
feature_level = (
    "morpheme",
    {
        "embed_target_embed": tune.grid_search([128, 256, 512]),
    },
    tokenize_into_morphemes,
    lambda config, dset: EmbedSingletonFeature(dset, config["embed_target_embed"])
)

split, split_name, epochs = splits
model_name, mk_model = model
(feature_name, _, extract_features, embed_features) = feature_level

name = f"split-{split_name}feature-{feature_level}_model-{model_name}"


cfg = {
    'lr': 0.00019401697177446437,
    'weight_decay': 9.230833759168172e-07,
    'hidden_dim': 128,
    'dropout': 0.1,
    'batch_size': 1,
    'epochs': 30,
    'gradient_clip': 1,
    'embed_target_embed': 256
}

for lang in ["ZU"]:
    print(f"Training {split_name}-level, {feature_name}-feature {model_name} for {lang}")
    train_class, _ = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features, map_tag=classes_only_no_tags)
    train_tag, _ = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features, map_tag=tags_only_no_classes)
    train, valid = AnnotatedCorpusDataset.load_data(lang, split=split, tokenize=extract_features)

    m = model_for_config(mk_model, embed_features, train, cfg)
    m.tag_tagger = torch.load("out_models/bilstm-tags-ZU.pt")
    m.class_tagger = torch.load("out_models/bilstm-classes-ZU.pt")
    m.tag_ix_to_tag = train_tag.ix_to_tag
    m.class_ix_to_class = train_class.ix_to_tag

    _, _, report, _, _, _ = analyse_model(
        m, cfg, valid
    )

    print(report)
