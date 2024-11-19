# TODO: Use raytune
from amaterasu.model import setup_model
from amaterasu.preprocessing import preprocess_data
from amaterasu.train import read_config


def finetune():
    """Attempts to find the best hyperparameters for the model"""
    config = read_config()

    corpora, ngram_embeddings, (train_loader, validate_loader, test_loader) = preprocess_data(config)
    model, optimizer, criterion, scheduler = setup_model(config, corpora[0])

    parameters_ranges = {
        "dropout": (0.2, 0.5),
        "learning_rate": (0.1, 0.001),
        "batch_size": (32, 256),
        "weight_decay": (0.1, 0.0001),
        "t_0": (10, 20),
        "t_mult": (1.2, 2.5)
    }