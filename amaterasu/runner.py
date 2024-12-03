import torch

from amaterasu.aliases import Config
from amaterasu.model import setup_prod_model
from amaterasu.preprocessing import load_ngram_embeddings, sentence_to_vectors, LABEL_LIST
from amaterasu.train import read_config

import MeCab

def preprocess_data(config: Config, sentences: list[str]):
    ngram_embeddings = load_ngram_embeddings(config.embeddings_path, config.embedding_dim)
    vector = torch.tensor(sentence_to_vectors(config, ngram_embeddings, sentences[0]))

    return vector

def resolve_morpheme_boundaries(sentence: str, labels: list[str]):
    tokenized_sentence = ""
    for i, character in enumerate(sentence):
        # Add a space before the character if B or S and no space was added before
        if i > 0 and labels[i] in ["B", "S"] and tokenized_sentence[i-1] != " ":
            tokenized_sentence += " "

        tokenized_sentence += character

        # Add a space after the character if E or S
        if i < len(sentence) - 1 and labels[i] in ["E", "S"]:
            tokenized_sentence += " "

    return tokenized_sentence


def tokenize(sentences: list[str]):
    config = read_config()
    model = setup_prod_model(config)
    input_layer = preprocess_data(config, sentences)

    model.eval()
    predictions = torch.argmax(model(input_layer), dim=1)
    predictions = [LABEL_LIST[x] for x in predictions]

    print(predictions)
    print(resolve_morpheme_boundaries(sentences[0], predictions))

if __name__ == "__main__":
    tokenize(["日本語の文書を分割してみましょう"])
