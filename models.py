import torch
import torch.nn as nn
import torch.nn.functional as F

"Simple DNN"
class DNN(nn.Module):
    def __init__(self,num_classes=3):
        super(DNN,self).__init__()
        self.fc1 = nn.Linear(2548,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,num_classes)
    def forward(self,x):
        x = F.dropout(F.relu(self.fc1(x)),0.5)
        x = F.dropout(F.relu(self.fc2(x)),0.5)
        x = F.dropout(F.relu(self.fc3(x)),0.5)
        x = self.fc4(x)
        return x
"Reference: https://iopscience.iop.org/article/10.1088/1742-6596/2024/1/012044/pdf"
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=80, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=80, out_channels=160, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=160, out_channels=320, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(320 * 319, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.dropout(F.relu(self.conv1(x)), 0.4)
        x = F.dropout(F.relu(self.conv2(x)), 0.4)
        x = F.dropout(F.relu(self.conv3(x)), 0.4)
        x = F.dropout(F.relu(self.conv4(x)), 0.4)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
"Reference: https://aircconline.com/csit/papers/vol11/csit112328.pdf"
class CNN_GRU(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_GRU, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3) 
        self.pool1 = nn.MaxPool1d(kernel_size=2) 
        self.dropout1 = nn.Dropout(p=0.2)  
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)  
        self.pool2 = nn.MaxPool1d(kernel_size=2)  
        self.dropout2 = nn.Dropout(p=0.2)  
        
        self.gru1 = nn.GRU(input_size=128, hidden_size=256, batch_first=True) 
        self.dropout3 = nn.Dropout(p=0.2)  
        self.gru2 = nn.GRU(input_size=256, hidden_size=32, batch_first=True) 
        self.dropout4 = nn.Dropout(p=0.2)  
        
        self.fc1 = nn.Linear(32, 128) 
        self.fc2 = nn.Linear(128, num_classes)  

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = F.relu(self.conv1(x))  
        x = self.pool1(x)  
        x = self.dropout1(x)  
        
        x = F.relu(self.conv2(x))  
        x = self.pool2(x) 
        x = self.dropout2(x)  
        
        x = x.permute(0, 2, 1)  
        
        # GRU layers
        x, _ = self.gru1(x) 
        x = self.dropout3(x)  
        x, _ = self.gru2(x)  
        x = self.dropout4(x) 
        
        x = x[:, -1, :] 
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        
        return x
"Reference: https://aircconline.com/csit/papers/vol11/csit112328.pdf"
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN_LSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.2) 
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3) 
        self.pool2 = nn.MaxPool1d(kernel_size=2)  
        self.dropout2 = nn.Dropout(p=0.2)  
        
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, num_layers=1) 
        self.dropout3 = nn.Dropout(p=0.2)  
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=32, batch_first=True, num_layers=1) 
        self.dropout4 = nn.Dropout(p=0.2) 
        
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x)) 
        x = self.pool1(x) 
        x = self.dropout1(x) 
        
        x = F.relu(self.conv2(x)) 
        x = self.pool2(x)  
        x = self.dropout2(x) 
        
        x = x.permute(0, 2, 1)  
        
        x, _ = self.lstm1(x)  
        x = self.dropout3(x)  
        x, _ = self.lstm2(x) 
        x = self.dropout4(x)  
        
        x = x[:, -1, :]  
        
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)  
        
        return x
"Reference: https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2020.00043"
class CNN_SAE_DNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN_SAE_DNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        cnn_output_size = self._get_cnn_output_size(2548)
        
        self.encoder = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, cnn_output_size),
            nn.ReLU()
        )
        
        self.decoder_output_to_dnn_input = nn.Linear(cnn_output_size, 256)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def _get_cnn_output_size(self, input_size):
        x = torch.randn(1, 1, input_size)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        return x.flatten().shape[0]
    
    def forward(self, x):
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        x = self.decoder_output_to_dnn_input(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
