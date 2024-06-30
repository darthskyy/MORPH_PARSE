import warnings
from utils import MorphParseArgs, MorphParseDataset, MorphParseModel
from transformers import XLMRobertaTokenizerFast
import time

def main():
    # parsing the hyperparameters for the 

    # disable the warnings
    warnings.filterwarnings("ignore")
    # getting the Runner arguments for the program
    args = MorphParseArgs()

    experiment_id = f"{args['language']}_{args['model_dir'].split('/')[-1]}" + time.strftime("_%Y%m%d_%H%M%S")
    # loading the tokenizer and the dataset
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args["model_dir"], cache_dir=args["cache_dir"])
    
    dataset = MorphParseDataset(
        language=args["language"], tokenizer=tokenizer, path=args["data_dir"], seed=args["seed"])
    dataset.load()
    dataset.tokenize()
    
    model = MorphParseModel(
        language=args["language"], path=args["model_dir"], tokenizer=tokenizer, dataset=dataset)
    model.load()

    # grid search for hyperparameters
    hyperparameters = {
        "learning_rate": [1e-5, 2e-5, 3e-5],
        "num_train_epochs": [3, 5, 10],
        "per_device_train_batch_size": [8, 16, 32]
    }

    for epochs in hyperparameters["num_train_epochs"]:
        for lr in hyperparameters["learning_rate"]:
            for batch_size in hyperparameters["per_device_train_batch_size"]:
                model.args.learning_rate = lr
                model.args.num_train_epochs = epochs
                model.args.per_device_train_batch_size = batch_size
                print(model)
                start = time.time()
                model.train()
                results = model.evaluate_test()
                runtime = float(time.time() - start)/3600
                # writes results in the form
                # language{sep}model{sep}lr{sep}epochs{sep}batch_size{sep}loss{sep}f1{sep}precision{sep}recall{sep}runtime
                # enters the macro-averaged F1, precision, and recall scores
                with open(args["log_file"], "a") as f:
                    print(f"{args['language']},{args['model_dir']},{model.args.learning_rate},{model.args.num_train_epochs},{model.args.per_device_train_batch_size},{results['eval_macro-f1']},{results['eval_macro-precision']},{results['eval_macro-recall']},{runtime:.4f}", file=f)
                
    # model.train()

if __name__ == "__main__":
    main()