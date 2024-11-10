from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from amaterasu.aliases import Config


class Amaterasu(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate if n_layers > 1 else 0)
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


def compute_class_weights(device: torch.device, corpus):
    label_counts = defaultdict(int)

    for sentence in corpus:
        for char, label in zip(sentence['characters'], sentence['labels']):
            label_counts[label] += 1

    class_counts = torch.tensor([label_counts['S'], label_counts['B'], label_counts['E'], label_counts['I']],
                                dtype=torch.float).to(device)
    class_weights = 1.0 / class_counts

    class_weights = class_weights / class_weights.sum()

    pad_weight = torch.zeros(1).to(device)
    class_weights = torch.cat((pad_weight, class_weights))

    return class_weights

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)




def setup_model(config: Config, corpus) -> tuple[Amaterasu, optim.AdamW, nn.CrossEntropyLoss, optim.lr_scheduler.CosineAnnealingLR]:
    model = Amaterasu(config.input_dim, config.hidden_dim, config.output_dim, config.layers, config.bidirectional,
                      config.dropout).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(config.device, corpus)).to(config.device)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, "min")

    print(f'The model has {count_parameters(model):,} learnable parameters')

    return model, optimizer, criterion, scheduler
