import argparse
import torch
from data_loader import DataLoaderModule
from models import DNN, CNN, CNN_GRU, CNN_LSTM, CNN_SAE_DNN 
from train import train_model, evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['dnn', 'cnn', 'cnn_gru', 'cnn_lstm', 'cnn_sae_dnn'], required=True, help='Model to train (dnn, cnn, etc.)')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=70, help='Number of epochs')
    
    args = parser.parse_args()

    data_loader = DataLoaderModule(args.data, batch_size=args.batch_size)
    train_loader, test_loader = data_loader.get_loaders()

    model_name = args.model
    if args.model == 'dnn':
        model = DNN()
    elif args.model == 'cnn':
        model = CNN()
    elif args.model == 'cnn_gru':
        model = CNN_GRU()
    elif args.model == 'cnn_lstm':
        model = CNN_LSTM()
    elif args.model == 'cnn_sae_dnn':
        model = CNN_SAE_DNN()
    else:
        raise ValueError("Invalid model type")

    train_model(model, train_loader, test_loader, args.epochs, args.lr, model_name)

if __name__ == '__main__':
    main()
