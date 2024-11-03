import torch

class Hyperparameters:
  def __init__(self, n_epochs, batch_size, learning_rate):
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.window_size = 3
    self.device = device = torch.device('cuda') if torch.cuda and torch.cuda.is_available() else torch.device('cpu')