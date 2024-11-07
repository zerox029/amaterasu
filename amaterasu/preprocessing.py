import time

import torch
from torch.utils.data import random_split, DataLoader

import bcolz
from bcolz.carray_ext import carray

import nltk
import pickle
import numpy as np
from numpy import ndarray

import random
import re
from decimal import Decimal
from pathlib import Path

from aliases import Corpus, CharacterType, NGramEmbeddings, Config

PAD_TOKEN: str = "<pad>"

class NGram:
    def __init__(self, n: int, characters: list[str], dimensionality: int, embeddings: NGramEmbeddings):
        self.n: int = n
        self.characters: list[str] = characters
        self.embedding: ndarray = self._get_embedding(dimensionality, embeddings)

    def _get_embedding(self, dimensionality: int, ngram_embeddings: NGramEmbeddings) -> ndarray:
        ngram_str: str = ''.join(self.characters)

        ngram_embedding: ndarray = np.zeros(dimensionality)
        if ngram_str in ngram_embeddings:
            ngram_embedding: ndarray = np.array([Decimal(x) for x in ngram_embeddings[ngram_str]])

        return ngram_embedding


class CharacterVector:
    def __init__(self, unigram: NGram, bigram: NGram, trigram: NGram):
        self.unigram: NGram = unigram
        self.bigram: NGram = bigram
        self.trigram: NGram = trigram
        self.characters: tuple[str] = self.trigram.characters
        self.embedding = self._get_embeddings()

    def _get_embeddings(self) -> ndarray:
        character_type_embeddings: ndarray = np.zeros(len(self.characters) * len(CharacterType))
        for i, character in enumerate(self.characters):
            character_type_embeddings[get_character_type(character).value + (len(CharacterType) * i)] = 1

        return np.concatenate((self.unigram.embedding,
                               self.bigram.embedding,
                               self.trigram.embedding,
                               character_type_embeddings))


def setup_corpora() -> tuple[Corpus, Corpus]:
    """
    Sets up KNBC and JEITA corpora to be used by the model
    """

    nltk.download('knbc')
    nltk.download('jeita')

    knbc = nltk.corpus.knbc.sents()
    jeita = nltk.corpus.jeita.sents()

    return _setup_corpus(knbc), _setup_corpus(jeita)


def _setup_corpus(raw_corpus) -> Corpus:
    """Sets up a single corpus to be used by the model"""

    tagged_corpus: Corpus = []

    for sentence in raw_corpus:
        tagged_sentence: dict[str, list[str]] = {'characters': [], 'labels': []}

        for word in sentence:
            for position, char in enumerate(word):
                tagged_sentence['characters'].append(char)
                if len(word) == 1:
                    tagged_sentence['labels'].append('S')
                elif position == 0:
                    tagged_sentence['labels'].append('B')
                elif position == len(word) - 1:
                    tagged_sentence['labels'].append('E')
                else:
                    tagged_sentence['labels'].append('I')

        tagged_corpus.append(tagged_sentence)

    return tagged_corpus


def load_ngram_embeddings(embeddings_path: str) -> NGramEmbeddings:
    """Loads ngram embeddings from the given embeddings file and returns them in a dict."""

    tmp_dir = "data/tmp"

    if (not Path(f'{tmp_dir}/200D.dat').is_dir()) or (not Path(f'{tmp_dir}/200D.pkl').is_file()) or (not Path(f'{tmp_dir}/200D_idx.pkl').is_file()):
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        ngrams: list[str] = []
        idx: int = 0
        ngram2idx: dict[str, int] = {}
        vectors: carray = bcolz.carray(np.zeros(1), rootdir=f'{tmp_dir}/200D.dat', mode='w')

        with open(embeddings_path, 'rb') as f:
            next(f)  # Skip the matrix size indicator
            for l in f:
                line: list[str] = l.decode().split()

                if idx >= 2000:
                    break

                # This might not be necessary, but the embeddings file contained malformed lines at some point
                if len(line) != 200 + 1:
                    continue

                word: str = line[0]
                ngrams.append(word)
                ngram2idx[word] = idx

                idx += 1
                vect: ndarray = np.array(line[1:], dtype=float)
                vectors.append(vect)

        vectors: carray = bcolz.carray(vectors[1:].reshape((2000, 200)), rootdir=f'{tmp_dir}/200D.dat', mode='w')
        vectors.flush()
        pickle.dump(ngrams, open(f'{tmp_dir}/200D.pkl', 'wb'))
        pickle.dump(ngram2idx, open(f'{tmp_dir}/200D_idx.pkl', 'wb'))

    vectors: ndarray = bcolz.open(f'{tmp_dir}/200D.dat')[:]
    ngrams = pickle.load(open(f'{tmp_dir}/200D.pkl', 'rb'))
    ngram2idx = pickle.load(open(f'{tmp_dir}/200D_idx.pkl', 'rb'))

    return {n: vectors[ngram2idx[n]] for n in ngrams}


def get_character_type(char: str) -> CharacterType:
    """Returns the matching character type for a given character"""

    if re.match(u'[ぁ-ん]', char):
        return CharacterType.HIRAGANA
    elif re.match(u'[ァ-ン]', char):
        return CharacterType.KATAKANA
    elif re.match(u'[一-龥]', char):
        return CharacterType.KANJI
    elif re.match(u'[A-Za-z]', char):
        return CharacterType.ROMAJI
    elif re.match(u'[0-9０-９]', char):
        return CharacterType.DIGIT
    else:
        return CharacterType.OTHER


def create_loaders(corpus: Corpus, config: Config, ngram_embeddings: NGramEmbeddings) -> tuple[DataLoader, DataLoader, DataLoader]:
    def _collate_fn(batch):
        max_sentence_length = max([len(sentence['characters']) for sentence in batch])
        padded_sentences = np.zeros(shape=(len(batch), max_sentence_length))
        padded_labels = np.zeros(shape=(len(batch), max_sentence_length))


    def collate_fn(batch):
        # TODO: This is way too slow, should be improved dramatically
        max_sentence_length = max([len(sentence['characters']) for sentence in batch])
        padded_characters = np.zeros(shape=(len(batch), max_sentence_length))
        padded_labels = np.zeros(shape=(len(batch), max_sentence_length))
        start_time = time.time()

        for i, sentence in enumerate(batch):
            characters = sentence['characters']
            labels = sentence['labels']

            padded_sentence = characters + [PAD_TOKEN] * (max_sentence_length - len(characters))
            padded_character_vectors = generate_character_vectors(padded_sentence, config.window_size, ngram_embeddings)

            print(f"Generated padded: {time.time() - start_time}")

            character_vector_embeddings = np.zeros(len(padded_character_vectors))
            for j, character_vector in enumerate(padded_character_vectors):
                character_vector_embeddings[j] = character_vector.embedding

            print(f"Generated embeddings: {time.time() - start_time}")

            padded_characters[i] = character_vector_embeddings

            padded_sentence_labels = labels + [PAD_TOKEN] * (max_sentence_length - len(labels))
            padded_labels[i] = generate_label_vectors(padded_sentence_labels)

            print(f"Done: {start_time}")

        return (torch.tensor(np.array(padded_characters), dtype=torch.float32).to(config.device),
                torch.tensor(padded_labels, dtype=torch.float32).to(config.device))

    train_data, valid_data, test_data = random_split(corpus, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_data, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=config.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, collate_fn=collate_fn)

    return train_loader, validate_loader, test_loader


def generate_character_vectors(sentence: list[str], window_size: int, ngram_embeddings) -> ndarray[CharacterVector]:
    """Generates character vectors from a given sentence"""
    vector_count: int = len(sentence)

    character_vectors = np.zeros(shape=vector_count, dtype=CharacterVector)
    for t in range(0, vector_count):
        characters = sentence[t:t + window_size]

        character_1 = characters[0] if len(characters) > 0 else PAD_TOKEN
        character_2 = characters[1] if len(characters) > 1 else PAD_TOKEN
        character_3 = characters[2] if len(characters) > 2 else PAD_TOKEN

        unigram = NGram(1, [character_1], 200, ngram_embeddings)
        bigram = NGram(2, [character_1, character_2], 200, ngram_embeddings)
        trigram = NGram(3, [character_1, character_2, character_3], 200, ngram_embeddings)

        character_vectors[t] = CharacterVector(unigram, bigram, trigram)

    return character_vectors


def generate_label_vectors(labels) -> ndarray:
    label_vectors = np.zeros(len(labels))
    label_list = [PAD_TOKEN, 'S', 'B', 'E', 'I']

    for i, label in enumerate(labels):
        label_vector = np.zeros(len(label_list))
        label_vector[label_list.index(label)] = 1
        label_vectors[i] = label_vector

    return label_vectors


def preprocess_data(config: Config) -> tuple[tuple[Corpus, Corpus], NGramEmbeddings, tuple[DataLoader, DataLoader, DataLoader]]:
    if config.set_seed:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

    knbc, jeita = setup_corpora()
    ngram_embeddings = load_ngram_embeddings(config.embeddings_path)
    loaders = create_loaders(knbc, config, ngram_embeddings)

    return (knbc, jeita), ngram_embeddings, loaders
