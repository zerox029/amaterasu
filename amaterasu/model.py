from configparser import SectionProxy

import torch
import torch.nn as nn
import torch.optim as optim


class Amaterasu(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout_rate if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, text):
        outputs, _ = self.lstm(text)
        predictions = self.log_softmax(self.fc(self.dropout(outputs)))

        return predictions

    def initialize_weights(self):
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def compute_class_weights(device: torch.device):
    class_weights = torch.tensor([1, 1, 1, 1, 2], dtype=torch.float).to(device)

    return class_weights

def setup_model(hyperparameters: SectionProxy) -> tuple[Amaterasu, optim.AdamW, nn.CrossEntropyLoss]:
    embedding_dim = hyperparameters.getint('EmbeddingDim')
    window_size = hyperparameters.getint('WindowSize')
    input_dim = embedding_dim * window_size + 6 * window_size
    hidden_dim = hyperparameters.getint('HiddenDim')
    n_layers = hyperparameters.getint('NLayers')
    bidirectional = hyperparameters.getboolean('Bidirectional')
    dropout_rate = hyperparameters.getfloat('DropoutRate')
    learning_rate = hyperparameters.getfloat('LearningRate')
    device = torch.device('cuda') if torch.cuda and torch.cuda.is_available() else torch.device('cpu')

    model = Amaterasu(input_dim, hidden_dim, hidden_dim, n_layers, bidirectional, dropout_rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(device)).to(device)

    return model, optimizer, criterion