# Morph Parse PLM

This project provides tools for training a morphological parser using the utilities provided in `utils.py`.

## Preparations
### Prerequisites

Ensure you have the following installed:
- Python 3.10+
- Required Python packages (listed in `requirements.txt`)
- CUDA 11.5.1

Install the required packages using:
```sh
pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```

### Acquiring the project files
The main required files to train the models developed in this project are [utils.py](./utils.py), [train.py](./train.py) and [data_prep.py](../scripts/data_prep.py). They can just be downloaded from this repository and put into a project folder containing the same structure:

project_folder/ <br>
├── plm/ <br>
├ ├── utils.py<br>
├ ├── train.py<br>
├── scripts/ <br>
├ ├── data_prep.py<br>
└──────────── <br>

This part of the project can be clone using the command in project_folder:
    ```sh
    git clone --branch plm https://github.com/darthskyy/MORPH_PARSE .
    ```

## Running Models
### Training the Model

To train the model, follow these steps (all steps assume you are in the parent directory for the project):

1. **Prepare your dataset**: Ensure that the dataset are prepared in the TSV form that they are expected for the model. The directory containing the data must have two subdirectories: TEST and TRAIN containing the respective files (some modification of the utils.MorphParseDataset in the [utils.py](./utils.py) file to read in a specific VALID set if there is one otherwise it is derived as 10% of the TRAIN file). In the respective dataset there must be the files for the specific language that you want to train; for example if training for isiNdebele (NR), the files NR_TRAIN.tsv and NR_TEST.tsv must be in their respective subdirectories. <br>
If you do not have the data, you can run the [data prep script](../scripts/data_prep.py) to download the files and prepare them into the require format. Use the command:
    ```sh
    python3 scripts/data_prep.py
        --output_dir path/to/save/data_files \
        --url source_url/with/sadilar/zip/file (optional already set by default) \
        --clean (optional to remove unnecessary files for training)
    ```

2. **Run the training script**: Use the following command to start training the model:
    ```sh
    python3 plm/train.py --output_dir path/to/save/model \
        --model_dir path/to/saved/model/checkpoint \
        --data_dir path/to/data \
        --log_file log_file_name \
        --language language_code \
        --cache_dir path/to/huggingface/cache_for/efficiency \
        --seed seed
    ```

    Replace the command-line arguments with your specific directories and any saved checkpoints for models. There are other command-line arguments which are specified in the [Hugging Face Trainer class](https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/trainer#trainer). The specific arguments are ```num_train_epochs```, ```per_device_train_batch_size```, ```per_device_eval_batch_size```, ```learning_rate```, ```weight_decay```, ```eval_strategy```, ```eval_steps```, ```save_strategy```, ```eval_steps```, ```save_total_limit```,  ```load_best_model_at_end```,  ```metric_for_best_model```,  ```greater_is_better```, ```logging_steps```, and ```disable_tqdm```. They all work as specified in the class documentation and if not provided are set to default values as specified in the [utils.MorphParseModel](./utils.py) class.

### Example

Here is an example command to train the model:
```sh
python3 plm/train.py --data_dir data \
    --model_dir xlm-roberta-base \
    --output_dir morph_parse_NR \
    --language NR \
    --cache_dir .cache \
    --seed 42 --num_train_epochs 10 --logging_steps 1000 \
    --per_device_train_batch_size 32 --learning_rate 5e-5 \
    --save_total_limit 2 --metric_for_best_model macro_f1 \
    --greater_is_better
```

This command will train the model using `train_data.csv` and save the trained model in the `models` directory.

### Additional Options

For more options and detailed usage, refer to the help command:
```sh
python train.py --help
```

## Reproducing Experiments
### Grid Search
To perform a grid search for hyperparameter tuning, follow these steps:

1. **Run the grid search**:
This creates a results file configured as json file (if it doesn't already exist) which contains each configuration of the grid search based on the various combinations of the hyperparameters.
Execute the grid search script to start the hyperparameter tuning process:
```sh
python3 plm/grid_search.py --suffix <suffix> --results_file <target results file>
```
This script will iterate through all configurations, train the model, and evaluate its performance. The results will be saved in the specified results file.

2. **Monitor the progress**: The script will print the current configuration being processed and update the results file with the performance metrics of each configuration. You can monitor the progress by checking the console output and the results file. To monitor the progress run the [grid_summary](../scripts/grid_summary.py):
```sh
python3 scripts/grid_summary.py --results_file <grid search results file>
    --save_csv <optional_file_to_export_to_csv>
```

3. **Analyze the results**: Once the grid search is complete, analyze the results stored in the results file to determine the best hyperparameter combination based on the evaluation metrics.

#### Example

Here is an example command to run the grid search:
```sh
python3 plm/grid_search.py --suffix _SURFACE --results_file results/grid_surface.json
python3 scripts/grid_summary.py --results_file results/grid_surface.json
```

This command will generate and evaluate all possible configurations, storing the results in `results/grid_search_results.json`.

For more options and detailed usage, refer to the help command:
```sh
python3 plm/grid_search.py --help
```
### Final Testing

To perform final testing of the trained models, the pre-requisite is having done the grid search thoroughly to discern the best model each language-model pair

1. **Run the evaluation script**: Use the following command to start the evaluation of the models:
```sh
python3 plm/test_eval.py --suffix <suffix>
    --results_file <final test results output file>
    --grid_search_file <grid search input file>
```

Replace the command-line arguments with your specific directories and files. The script will iterate through the configurations, evaluate the models, and update the results file with the performance metrics. The script will also save the predictions produced from these run

2. **Monitor the progress**: The evaluation script will print the current configuration being processed and update the results file with the performance metrics of each configuration. To monitor the progress you run the [test_summary](../scripts/test_summary.py) scriptn which shows which seeds were run.

#### Example

Here is an example command to run the final testing:
```sh
python3 plm/test_eval.py --suffix _SURFACE --results_file results/final_results_surface.json --grid_search_file results/grid_surface.json
python3 scripts/test_summary.py --results_file results/final_results_surface.json
```

This command will evaluate the models using the configurations specified in the `results/config.json` file and save the results in `results/final_results.json`.

For more options and detailed usage, refer to the help command:
```sh
python3 plm/test_eval.py --help
```

## Contributions
For any questions or issues, please open an issue on the [GitHub repository](https://github.com/darthskyy/MORPH_PARSE).
