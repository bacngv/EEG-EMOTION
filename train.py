import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
import os 

def train_model(model, train_loader, test_loader, num_epochs, lr, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        loss_running = 0.0
        for inputs, labels in train_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
                model.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_running += loss.item()

        test_acc = test_accuracy(model, test_loader)
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss_running / len(train_loader):.4f}, Test Accuracy: {test_acc:.2f}%')

    evaluate_and_save_report(model, test_loader, model_name)

def test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def evaluate_and_save_report(model, test_loader, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Ensure the outputs directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    output_file = os.path.join(output_dir, f"{model_name}_classification_report.csv")
    report_df.to_csv(output_file)
    print(f'Classification report saved to {output_file}')

    print(report_df)

def evaluate_model(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test).sum().item()
        total = len(y_test)
        model_acc = correct / total * 100
        print("Test Accuracy: {:.3f}%".format(model_acc))

    return predicted