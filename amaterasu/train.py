import configparser
import logging
import sys
from datetime import datetime
from pathlib import Path

import math
import time

import numpy as np
import torch
from tabulate import tabulate

from amaterasu.aliases import Config
from amaterasu.metrics import categorical_accuracy, f1, compute_metrics, generate_confusion_matrix
from amaterasu.model import setup_model, Amaterasu
from amaterasu.preprocessing import preprocess_data, LABEL_LIST

logger = logging.getLogger(__name__)


def read_config() -> Config:
    config = configparser.ConfigParser()

    config.read('config.ini', encoding='utf-8')

    default_config = config["GENERAL"]
    set_seed = default_config.getboolean('SetSeed')
    seed = default_config.getint('Seed')
    embeddings_path = default_config.get('EmbeddingsPath')
    custom_dataset_path = default_config.get('CustomDatasetPath')
    window_size = default_config.getint('WindowSize')

    hyperparameters_config = config['HYPERPARAMETERS']
    embedding_dim = hyperparameters_config.getint('EmbeddingDim')
    input_dim = embedding_dim * window_size + 6 * window_size
    hidden_dim = hyperparameters_config.getint('HiddenDim')
    n_layers = hyperparameters_config.getint('NLayers')
    bidirectional = hyperparameters_config.getboolean('Bidirectional')
    dropout_rate = hyperparameters_config.getfloat('DropoutRate')
    learning_rate = hyperparameters_config.getfloat('LearningRate')
    n_epochs = hyperparameters_config.getint('NEpochs')
    batch_size = hyperparameters_config.getint('BatchSize')

    config = Config(set_seed, seed, embeddings_path, custom_dataset_path, window_size, embedding_dim,
                    input_dim, hidden_dim, n_layers, bidirectional, dropout_rate,
                    learning_rate, n_epochs, batch_size)

    return config


def train_single_epoch(model: Amaterasu, optimizer, criterion, loader, start_time, epoch, n_epochs, batch_length):
    epoch_loss = 0
    epoch_accuracy = 0

    model.train()

    for i, (sentence, labels) in enumerate(loader):
        display_epoch_status(i, batch_length, start_time, epoch, n_epochs, "training")
        optimizer.zero_grad()

        predictions = model(sentence)

        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1, labels.shape[-1])

        loss = criterion(predictions, labels)
        accuracy, (average_f1, categorical_f1) = compute_metrics(predictions, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += accuracy

        logger.info(
            f"Epoch {epoch} - Minibatch {i + 1}: Loss {epoch_loss / (i + 1)} - Accuracy {epoch_accuracy / (i + 1)} (Learning rate {optimizer.param_groups[0]['lr']})")

    return epoch_loss / len(loader), epoch_accuracy / len(loader)


def evaluate_single_epoch(model, criterion, loader, start_time, epoch, n_epochs, batch_length):
    epoch_loss = 0
    epoch_accuracy = 0

    model.eval()

    for i, (sentence, labels) in enumerate(loader):
        display_epoch_status(i + batch_length - len(loader), batch_length, start_time, epoch, n_epochs, "evaluating")

        predictions = model(sentence)

        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1, labels.shape[-1])

        loss = criterion(predictions, labels)
        accuracy, (average_f1, categorical_f1) = compute_metrics(predictions, labels)

        epoch_loss += loss.item()
        epoch_accuracy += accuracy

        logger.info(
            f"Epoch {epoch} - Minibatch {i + 1}: Loss {epoch_loss / (i + 1)} - Accuracy {epoch_accuracy / (i + 1)}")

    return epoch_loss / len(loader), epoch_accuracy / len(loader)


def test_batch(model, criterion, loader, start_time, batch_length):
    epoch_loss = 0
    epoch_accuracy = 0
    epoch_average_f1 = 0
    epoch_categorical_f1 = []

    batch_predictions = []
    batch_labels = []

    model.eval()

    for i, (sentence, labels) in enumerate(loader):
        display_epoch_status(i + batch_length - len(loader), batch_length, start_time, 0, 1, "testing")

        predictions = model(sentence)

        predictions = predictions.view(-1, predictions.shape[-1])
        batch_predictions.append(predictions)
        labels = labels.view(-1, labels.shape[-1])
        batch_labels.append(labels)

        loss = criterion(predictions, labels)
        accuracy, (average_f1, categorical_f1) = compute_metrics(predictions, labels)

        epoch_loss += loss.item()
        epoch_accuracy += accuracy
        epoch_average_f1 += average_f1
        epoch_categorical_f1.append(categorical_f1)

    generate_confusion_matrix(batch_predictions, batch_labels)

    epoch_categorical_f1 = np.array(epoch_categorical_f1).mean(axis=0)
    return epoch_loss / len(loader), epoch_accuracy / len(loader), epoch_average_f1 / len(loader), epoch_categorical_f1


def epoch_time(start_time: float, end_time: float):
    """Computes the elapsed time between two points in minutes and seconds"""

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def display_epoch_status(batch_progress: int, batch_length: int, start_time: float, epoch_number: int, n_epochs: int,
                         mode: str):
    """
    Prints the current epoch status for monitoring during training

    Args:
        batch_progress (int): The number of completed datapoints within the current batch
        batch_length (int): The total number of datapoints within the current batch
        start_time (float): The start time of the current epoch
        epoch_number (int): The current epoch number
        n_epochs (int): The total number of epochs
        mode (string): The current mode ("training" or "evaluating")
    """

    epoch_mins, epoch_secs = epoch_time(start_time, time.time())
    progress_percent = (batch_progress + 1) / batch_length
    progress_bar_fill = math.ceil(20 * progress_percent)

    print(f'\rEpoch: {epoch_number + 1:02}/{n_epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s | (Currently {mode}) ',
          end="")
    print(f"[{'=' * progress_bar_fill}{' ' * (20 - progress_bar_fill)}] {progress_percent * 100:.2f}%", end="")


def reset_model(model: Amaterasu):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def begin_training(resume_previous_training: bool = False):
    Path('data/logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=f'data/logs/trainlogs_{datetime.today().strftime('%Y%m%d')}.log', level=logging.INFO)

    config = read_config()

    corpus, ngram_embeddings, (train_loader, validate_loader, test_loader) = preprocess_data(config)
    model, optimizer, criterion, scheduler = setup_model(config, corpus)

    if resume_previous_training:
        model.load_state_dict(torch.load("data/model.pt"))

    best_validation_loss = float('inf')

    print(f"Starting training on {config.device}")

    train_loss_values = []
    validation_loss_values = []
    train_accuracy_values = []
    validation_accuracy_values = []
    for epoch in range(config.epochs):
        logger.info("Starting epoch {}".format(epoch + 1))
        start_time = time.time()

        train_loss, train_accuracy = train_single_epoch(model,
                                                        optimizer,
                                                        criterion,
                                                        train_loader,
                                                        start_time,
                                                        epoch,
                                                        config.epochs,
                                                        len(train_loader) + len(validate_loader))
        validation_loss, validation_accuracy = evaluate_single_epoch(model,
                                                                     criterion,
                                                                     validate_loader,
                                                                     start_time,
                                                                     epoch,
                                                                     config.epochs,
                                                                     len(train_loader) + len(validate_loader))

        scheduler.step(validation_loss)

        train_loss_values.append(train_loss)
        validation_loss_values.append(validation_loss)
        train_accuracy_values.append(train_accuracy.item() * 100)
        validation_accuracy_values.append(validation_accuracy.item() * 100)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), 'in_progress.pt')

        print(f'\rEpoch: {epoch + 1:02}/{config.epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
        print(f'\t Validation Loss: {validation_loss:.3f} |  Val. Acc: {validation_accuracy * 100:.2f}%')


def begin_testing(model_path: str):
    Path('data/logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=f'data/logs/testlogs_{datetime.today().strftime('%Y%m%d')}.log', level=logging.INFO)

    config = read_config()

    corpus, ngram_embeddings, (_, _, test_loader) = preprocess_data(config)
    model, optimizer, criterion, scheduler = setup_model(config, corpus)

    model.load_state_dict(torch.load(model_path))

    print(f"Starting testing on {config.device}")

    logger.info("__Starting testing__")

    start_time = time.time()

    loss, accuracy, average_f1_score, categorical_f1_score = test_batch(model,
                                                                        criterion,
                                                                        test_loader,
                                                                        start_time,
                                                                        len(test_loader))

    end_time = time.time()

    test_mins, test_secs = epoch_time(start_time, end_time)

    print(f"\nCompleted testing in {test_mins}m {test_secs}s")
    print(f"Loss: {loss:.3f} | Accuracy: {(accuracy * 100):.2f}% | Average F1: {average_f1_score:.3f}")
    print(f"F1 for class S: {categorical_f1_score[0]:.2f} | F1 for class B: {categorical_f1_score[1]:.2f} | F1 for class E: {categorical_f1_score[2]:.2f} | F1 for class I: {categorical_f1_score[3]:.2f}")


if __name__ == '__main__':
    if sys.argv[1] == 'test':
        begin_testing("data/model.pt")
    else:
        begin_training(True)
