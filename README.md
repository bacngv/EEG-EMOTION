# Emotion Recognition from EEG Signals

## Introduction

This project focuses on emotion recognition from EEG (electroencephalogram) signals using various deep learning models, including DNN (Deep Neural Network), CNN (Convolutional Neural Network), CNN-GRU, CNN-LSTM, and a hybrid model CNN-SAE-DNN. The EEG data is processed to classify emotions into three categories: NEGATIVE, NEUTRAL, and POSITIVE.

## Requirements

- Python 3.x
- Required libraries:
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

The project includes the following files:

- `data_loader.py`: This file contains the code for loading and processing the data.
- `visualization.py`: This file contains functions for visualizing the data.
- `models.py`: This file defines the different deep learning models used for emotion recognition:
  - `DNN`: A simple deep neural network model.
  - `CNN`: A convolutional neural network model.
  - `CNN_GRU`: A model combining CNN and GRU (Gated Recurrent Unit).
  - `CNN_LSTM`: A model combining CNN and LSTM (Long Short-Term Memory).
  - `CNN_SAE_DNN`: A hybrid model using CNN, Sparse Autoencoder, and DNN.
- `training.py`: This file contains functions for training and evaluating the models.
- `main.py`: The main file to run the project with input parameters from the command line.

## Usage

1. **Load the data**: Ensure that you have the `emotions.csv` data file containing EEG data in the appropriate format.

2. **Run the model**: Use the command line to run the model. You can choose any of the models listed above. Below is the syntax for running the DNN model:

```bash
python main.py --data 'path_to_data_file.csv' --model dnn --epochs 10
```

You can substitute `dnn` with `cnn`, `cnn_gru`, `cnn_lstm`, or `cnn_sae_dnn` to run the respective model:

```bash
python main.py --data 'path_to_data_file.csv' --model cnn --epochs 10
python main.py --data 'path_to_data_file.csv' --model cnn_gru --epochs 10
python main.py --data 'path_to_data_file.csv' --model cnn_lstm --epochs 10
python main.py --data 'path_to_data_file.csv' --model cnn_sae_dnn --epochs 10
```

### Parameters

- `--data`: Path to the CSV data file.
- `--model`: The model to use for training (`dnn`, `cnn`, `cnn_gru`, `cnn_lstm`, or `cnn_sae_dnn`).
- `--epochs`: Number of epochs for training (default is 10).
- `--sampling_rate`: Sampling rate for the EEG data (default is 256).

## Visualization

The project includes visualization functions for:

- Emotion distribution charts
- Time-series EEG data
- Power spectrum
- Correlation matrix

## Contact

If you have any questions or suggestions, please feel free to contact me via email: [your email] or visit [your LinkedIn].

---

**Note**: This project is in the development stage, so there may be some bugs or shortcomings. Any feedback is welcome!