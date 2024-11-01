# main.py
import argparse
from data_loader import load_data
from visualization import plot_pie_chart, plot_time_series, plot_power_spectral_density, plot_correlation_heatmap
from models import DNN, CNN
from training import train_model, evaluate_model

def main(args):
    data_loader = load_data(args.data)
    
    # Visualize data
    plot_pie_chart(data_loader.X)
    plot_time_series(data_loader.X, 0)  # Sample index
    plot_power_spectral_density(data_loader.X.iloc[0], args.sampling_rate)
    plot_correlation_heatmap(data_loader.X)

    # Model selection
    if args.model == 'dnn':
        model = DNN()
    elif args.model == 'cnn':
        model = CNN()
    else:
        raise ValueError("Model not recognized. Please choose 'dnn' or 'cnn'.")

    # Training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, data_loader.train_loader, criterion, optimizer, num_epochs=args.epochs)

    # Evaluation
    evaluate_model(model, data_loader.test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on EEG data.')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--model', type=str, choices=['dnn', 'cnn'], required=True, help='Model to use for training.')
    parser.add_argument('--sampling_rate', type=int, default=256, help='Sampling rate for EEG data.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')

    args = parser.parse_args()
    main(args)
