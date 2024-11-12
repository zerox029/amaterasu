import torch
from torch import float32
from torch.utils.data import random_split, DataLoader

import bcolz
from bcolz.carray_ext import carray

import nltk
import pickle
import numpy as np
from numpy import ndarray

import random
import re
from pathlib import Path

from aliases import CharacterType, Config, Corpus, NGramEmbeddings, TrainValidateTest
from amaterasu.aliases import CorpusOrigin

PAD_TOKEN: str = "<pad>"

def setup_corpora(corpus_origin: CorpusOrigin) -> Corpus:
    """
    Sets up KNBC and JEITA corpora to be used by the model
    """

    if corpus_origin == CorpusOrigin.KNBC:
        nltk.download('knbc')
        knbc = nltk.corpus.knbc.sents()

        return _setup_corpus(knbc)

    elif corpus_origin == CorpusOrigin.JEITA:
        nltk.download('jeita')
        jeita = nltk.corpus.jeita.sents()

        return _setup_corpus(jeita)
    else:
        raise NotImplemented

def _setup_corpus(raw_corpus) -> Corpus:
    """Sets up a single corpus to be used by the model"""

    tagged_corpus = []

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


def load_ngram_embeddings(embeddings_path: str, embedding_dimension: int) -> NGramEmbeddings:
    """Loads ngram embeddings from the given embeddings file and returns them in a dict."""

    tmp_dir = "data/tmp"

    if (not Path(f'{tmp_dir}/{embedding_dimension}D.dat').is_dir()) or (not Path(f'{tmp_dir}/{embedding_dimension}D.pkl').is_file()) or (not Path(f'{tmp_dir}/{embedding_dimension}D_idx.pkl').is_file()):
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        ngrams: list[str] = []
        idx: int = 0
        ngram2idx: dict[str, int] = {}
        vectors: carray = bcolz.carray(np.zeros(1), rootdir=f'{tmp_dir}/{embedding_dimension}D.dat', mode='w')

        with open(embeddings_path, 'rb') as f:
            next(f)  # Skip the matrix size indicator
            for l in f:
                line: list[str] = l.decode().split()
                # This might not be necessary, but the embeddings file contained malformed lines at some point
                if len(line) != embedding_dimension + 1:
                    continue

                if idx % 100000 == 0:
                    print(f"\rloaded {idx} embeddings", end="")

                word: str = line[0]
                ngrams.append(word)
                ngram2idx[word] = idx

                idx += 1
                vect: ndarray = np.array(line[1:], dtype=float)
                vectors.append(vect)

        vectors: carray = bcolz.carray(vectors[1:].reshape((1775303, embedding_dimension)), rootdir=f'{tmp_dir}/{embedding_dimension}D.dat', mode='w')
        vectors.flush()
        pickle.dump(ngrams, open(f'{tmp_dir}/{embedding_dimension}D.pkl', 'wb'))
        pickle.dump(ngram2idx, open(f'{tmp_dir}/{embedding_dimension}D_idx.pkl', 'wb'))

        print()

    vectors: ndarray = bcolz.open(f'{tmp_dir}/{embedding_dimension}D.dat')[:]
    ngrams = pickle.load(open(f'{tmp_dir}/{embedding_dimension}D.pkl', 'rb'))
    ngram2idx = pickle.load(open(f'{tmp_dir}/{embedding_dimension}D_idx.pkl', 'rb'))

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


def create_loaders(corpus: Corpus, config: Config, ngram_embeddings: NGramEmbeddings) -> TrainValidateTest:
    def collate_fn(batch):
        max_sentence_length = max([len(sentence['characters']) for sentence in batch])
        batch_inputs = torch.zeros(size=(len(batch), max_sentence_length, config.input_dim)).to(config.device)
        batch_labels = torch.zeros(size=(len(batch), max_sentence_length, config.output_dim)).to(config.device)

        pad_embedding = np.zeros(config.embedding_dim)
        label_list = [PAD_TOKEN, 'S', 'B', 'E', 'I']

        for sentence_id, sentence in enumerate(batch):
            characters = sentence['characters']
            labels = sentence['labels']

            for character_id in range(max_sentence_length):
                ngram_characters = characters[character_id:character_id + config.window_size]

                character_1 = ngram_characters[0] if len(ngram_characters) > 0 else PAD_TOKEN
                character_2 = ngram_characters[1] if len(ngram_characters) > 1 else PAD_TOKEN
                character_3 = ngram_characters[2] if len(ngram_characters) > 2 else PAD_TOKEN

                character_type_1 = get_character_type(character_1)
                character_type_2 = get_character_type(character_2)
                character_type_3 = get_character_type(character_3)

                label = labels[character_id] if len(labels) > character_id else PAD_TOKEN

                unigram_embedding = torch.tensor(ngram_embeddings.get(character_1, pad_embedding), dtype=float32).to(config.device)
                bigram_embedding = torch.tensor(ngram_embeddings.get(character_1 + character_2, pad_embedding), dtype=float32).to(config.device)
                trigram_embedding = torch.tensor(ngram_embeddings.get(character_1 + character_2 + character_3, pad_embedding), dtype=float32).to(config.device)

                character_type_embeddings = torch.zeros(size=(config.window_size * len(CharacterType),)).to(config.device)
                character_type_embeddings[character_type_1.value + (len(CharacterType) * 0)] = 1
                character_type_embeddings[character_type_2.value + (len(CharacterType) * 1)] = 1
                character_type_embeddings[character_type_3.value + (len(CharacterType) * 1)] = 2

                label_embedding = torch.zeros(config.output_dim)
                label_embedding[label_list.index(label)] = 1

                batch_inputs[sentence_id][character_id] = torch.cat((unigram_embedding.clone().detach(),
                                                                     bigram_embedding.clone().detach(),
                                                                     trigram_embedding.clone().detach(),
                                                                     character_type_embeddings.clone().detach()), dim=0)
                batch_labels[sentence_id][character_id] = label_embedding.clone().detach()

        return batch_inputs, batch_labels

    train_data, valid_data, test_data = random_split(corpus, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_data, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    validate_loader = DataLoader(valid_data, batch_size=config.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, collate_fn=collate_fn)


    return train_loader, validate_loader, test_loader

def preprocess_data(config) -> tuple[Corpus, NGramEmbeddings, TrainValidateTest]:
    if config.set_seed:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

    corpus = setup_corpora(CorpusOrigin.KNBC)
    ngram_embeddings = load_ngram_embeddings(config.embeddings_path, config.embedding_dim)
    loaders = create_loaders(corpus, config, ngram_embeddings)

    return corpus, ngram_embeddings, loaders
