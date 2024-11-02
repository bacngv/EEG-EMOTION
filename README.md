# Emotion Recognition from EEG Signals

## Introduction

This project aims to develop a robust system for emotion recognition from EEG (electroencephalogram) signals using advanced deep learning models. The implemented models include DNN (Deep Neural Network), CNN (Convolutional Neural Network), CNN-GRU (Gated Recurrent Unit), CNN-LSTM (Long Short-Term Memory), and a hybrid model CNN-SAE-DNN (Convolutional Sparse Autoencoder Deep Neural Network). The EEG data is processed to classify emotions into three categories: NEGATIVE, NEUTRAL, and POSITIVE.

## Requirements

### System Requirements

- **Python**: Version 3.x

### Required Libraries

You will need the following Python libraries:

- numpy
- pandas
- torch
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

```bash
pip install numpy pandas torch matplotlib seaborn scikit-learn
```

## Project Structure

The project comprises the following files and modules:

- **`data_loader.py`**: Contains code for loading and preprocessing the EEG data.
- **`visualization.py`**: Includes functions for visualizing data distributions and results.
- **`models.py`**: Defines the various deep learning models used for emotion recognition:
  - `DNN`: A simple deep neural network model.
  - `CNN`: A convolutional neural network model.
  - `CNN_GRU`: A model combining CNN and GRU.
  - `CNN_LSTM`: A model combining CNN and LSTM.
  - `CNN_SAE_DNN`: A hybrid model using CNN, Sparse Autoencoder, and DNN.
- **`training.py`**: Contains functions for training and evaluating the models.
- **`main.py`**: The entry point for running the project with command-line parameters.

## Usage

1. **Prepare the Data**: Ensure that you have the `emotions.csv` data file formatted correctly for EEG data.

2. **Run the Model**: Execute the following command in the terminal to run the desired model. Below is an example command for running the CNN-GRU model:

```bash
python ./EEG-EMOTION/main.py --model cnn_gru --data './EEG-EMOTION/data/emotions.csv' --batch_size 32 --lr 0.001 --epochs 50
```

### Command-Line Parameters

- `--data`: Path to the CSV data file containing EEG signals.
- `--model`: The model to be used for training (`dnn`, `cnn`, `cnn_gru`, `cnn_lstm`, or `cnn_sae_dnn`).
- `--batch_size`: Batch size for training (default is 64).
- `--lr`: Learning rate for the optimizer (default is 0.001).
- `--epochs`: Number of epochs for training (default is 10).

## Visualization

The project includes various visualization functions, enabling users to create:

- Emotion distribution charts
- Time-series visualizations of EEG data
- Power spectrum analysis
- Correlation matrices

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.