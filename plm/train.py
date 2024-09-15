from utils import MorphParseArgs, MorphParseDataset, MorphParseModel, GenUtils

from pprint import pprint
from seqeval.metrics import classification_report
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

    # load the dataset and tokenizer
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(MODEL_DIR, cache_dir=CACHE_DIR)
    dataset = MorphParseDataset(language=LANGUAGE, tokenizer=tokenizer, path=DATA_DIR, seed=SEED)
    dataset.load()
    dataset.tokenize()
    
    
    model = MorphParseModel(language=LANGUAGE, tokenizer=tokenizer, path=MODEL_DIR, dataset=dataset)
    model.load()

    # train the model
    model.args.num_train_epochs = 1
    model.train()
    model.save()

    # evaluate the model on the test set
    if torch.cuda.is_available():
        parser = pipeline("ner", model=model.model, tokenizer=dataset.tokenizer, device=0, batch_size=16)
    else:
        parser = pipeline("ner", model=model.model, tokenizer=dataset.tokenizer)
    print("Pipeline created")

    # evaluate the model on the test set
    print("Evaluating the model on the test set")
    def reverse_ids(ids):
        return [dataset.id2label[id] for id in ids]
    pos_pattern = re.compile(r'Pos\d')
    def remove_pos(tags):
        return [tag for tag in tags if not pos_pattern.match(tag)]

    # getting the morphemes, references, and predictions
    # morphemes are space separated so that the parser can process them
    morphemes = dataset.test.to_pandas()['morpheme'].apply(lambda x: ' '.join(x)).tolist()
    # references are converted from ids to labels and the Pos tags are removed as it often exists in a double tag
    references = dataset.test.to_pandas()['tag'].apply(lambda x: reverse_ids(x)).apply(remove_pos).tolist()
    # predictions are obtained from the parser and converted to labels
    predictions = parser(morphemes)
    predictions = [GenUtils.format_ner_results(p)[1] for p in predictions]

    # align the predictions and references if they are not the same length
    for i in range(len(predictions)):
        references[i] = [r for r in references[i] if "Dem"]
        predictions[i], references[i] = GenUtils.align_seqs(predictions[i], references[i])
        # add the # to the predictions and references because the classification report expects NER
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

    pprint(results)
    with open(LOG_FILE, 'w') as f:
        f.write(str(results))
if __name__ == '__main__':
    main()