from utils import MorphParseArgs, MorphParseDataset, MorphParseModel, GenUtils

from seqeval.metrics import classification_report

import logging
import sys
import re
from transformers import XLMRobertaTokenizerFast, pipeline
import torch
import warnings
warnings.filterwarnings("ignore")

def main():
    # get the command line arguments
    args = MorphParseArgs()
    OUTPUT_DIR = args.output_dir
    MODEL_DIR = args.model_dir
    DATA_DIR = args.data_dir
    LOG_FILE = args.log_file
    LANGUAGE = args.language
    CACHE_DIR = args.cache_dir
    SEED = args.seed

    # setting up logging
    logger = logging.getLogger(f"train_script_{LANGUAGE}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Language: {LANGUAGE}")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info(f"Seed: {SEED}")

    # load the dataset and tokenizer
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(MODEL_DIR, cache_dir=CACHE_DIR)
    dataset = MorphParseDataset(language=LANGUAGE, tokenizer=tokenizer, path=DATA_DIR, seed=SEED)
    dataset.load()
    dataset.tokenize()
    
    
    model = MorphParseModel(language=LANGUAGE, tokenizer=tokenizer, path=MODEL_DIR, dataset=dataset)
    model.load()

    # train the model
    if args.train:
        logger.info("Training the model")
        logger.info("Model hyperparameters")
        logger.info(model.args)
        model.train()
    model.save()

    # evaluate the model on the test set
    if torch.cuda.is_available():
        parser = pipeline("ner", model=model.model, tokenizer=dataset.tokenizer, device=0, batch_size=16)
    else:
        parser = pipeline("ner", model=model.model, tokenizer=dataset.tokenizer)
    logger.info("Evaluating the model on the test set")

    # evaluate the model on the test set
    logger.info("Evaluating the model on the test set")
    def reverse_ids(ids):
        return [dataset.id2label[id] for id in ids]

    # getting the morphemes, references, and predictions
    # morphemes are space separated so that the parser can process them
    morphemes = dataset.test.to_pandas()['morpheme'].apply(lambda x: ' '.join(x)).tolist()
    # references are converted from ids to labels and the Pos tags are removed as it often exists in a double tag
    references = dataset.test.to_pandas()['tag'].apply(lambda x: reverse_ids(x))

    # pos_pattern = re.compile(r'Pos\d')
    # def remove_pos(tags):
    #     return [tag for tag in tags if not pos_pattern.match(tag)]
    # references = references.apply(lambda x: remove_pos(x)) # uncomment this line if you want to remove the Pos tags
    
    references = references.tolist()
    # predictions are obtained from the parser and converted to labels
    predictions = parser(morphemes)
    predictions = [GenUtils.format_ner_results(p)[1] for p in predictions]
    morphemes = dataset.test.to_pandas()['morpheme'].apply(lambda x: '_'.join(x)).tolist()

    lines = ["morphemes\ttarget\tprediction\n"]
    # align the predictions and references if they are not the same length
    for i in range(len(predictions)):
        predictions[i], references[i] = GenUtils.align_seqs(predictions[i], references[i])
        # add the # to the predictions and references because the classification report expects NER

        # write the morphemes, references, and predictions to a file
        output = ""
        output += morphemes[i] + "\t"
        output += "_".join(references[i]) + "\t"
        output += "_".join(predictions[i]) + "\n"
        lines.append(output)

        predictions[i] = ["#" + p for p in predictions[i]]
        references[i] = ["#" + r for r in references[i]]
    results = classification_report(references, predictions, output_dict=True)

    results = {
        "micro_precision": results["micro avg"]["precision"],
        "micro_recall": results["micro avg"]["recall"],
        "micro_f1": results["micro avg"]["f1-score"] ,
        "macro_f1": results["macro avg"]["f1-score"],
        "macro_precision": results["macro avg"]["precision"],
        "macro_recall": results["macro avg"]["recall"],
    }

    logger.info("RESULTS")
    logger.info(f"Micro F1: {results['micro_f1']}")
    logger.info(f"Macro F1: {results['macro_f1']}")
    logger.info(f"Micro Precision: {results['micro_precision']}")
    logger.info(f"Macro Precision: {results['macro_precision']}")
    logger.info(f"Micro Recall: {results['micro_recall']}")
    logger.info(f"Macro Recall: {results['macro_recall']}")
    
    if args.predictions:
        logger.info(f"Writing predictions to {args.predictions}")
        with open(args.predictions, 'w') as f:
            f.writelines(lines)
if __name__ == '__main__':
    main()