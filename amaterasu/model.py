from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from amaterasu.aliases import Config, Corpus


class Amaterasu(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.lstm = nn.LSTM(config.input_dim, config.hidden_dim, num_layers=config.layers, bidirectional=config.bidirectional,
                            dropout=config.dropout if config.layers > 1 else 0)
        self.fc = nn.Linear(config.hidden_dim * 2 if config.bidirectional else config.hidden_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)
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


def compute_class_weights(device: torch.device, corpus) -> torch.Tensor:
    """
    Computes the relative weight of each class in a given corpus. These are then used as
    a multiplier within the weight function to combat class imbalance
    """
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

def setup_model(config: Config, corpus: Corpus) -> tuple[Amaterasu, optim.AdamW, nn.CrossEntropyLoss, optim.lr_scheduler.ReduceLROnPlateau]:
    model = Amaterasu(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(config.device, corpus)).to(config.device)
    scheduler = ReduceLROnPlateau(optimizer, "min")

    print(f'The model has {count_parameters(model):,} learnable parameters')

    return model, optimizer, criterion, scheduler
