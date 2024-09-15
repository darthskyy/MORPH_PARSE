# Morph Parse PLM

This project provides tools for training a morphological parser using the utilities provided in `utils.py`.

## Prerequisites

Ensure you have the following installed:
- Python 3.7+
- Required Python packages (listed in `requirements.txt`)

Install the required packages using:
```sh
pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```

## Training the Model

To train the model, follow these steps:

1. **Prepare your dataset**: Ensure your dataset is in the correct format as expected by the utilities in `utils.py`.

2. **Run the training script**: Use the following command to start training the model:
    ```sh
    python utils.py --train --data_path /path/to/your/dataset --output_path /path/to/save/model
    ```

    Replace `/path/to/your/dataset` with the path to your dataset and `/path/to/save/model` with the desired output directory for the trained model.

## Example

Here is an example command to train the model:
```sh
python utils.py --train --data_path ./data/train_data.csv --output_path ./models
```

This command will train the model using `train_data.csv` and save the trained model in the `models` directory.

## Additional Options

For more options and detailed usage, refer to the help command:
```sh
python utils.py --help
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/yourusername/morph_parse).
