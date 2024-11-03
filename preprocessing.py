import torch
from torch.utils.data import random_split

import nltk

import numpy as np

import bcolz
import pickle

import random
import itertools
import re
from collections import defaultdict
from decimal import Decimal

_SEED: int = 42
_SET_SEED: bool = True
PAD_TOKEN = "<pad>"

def setup_corpora():
    nltk.download('knbc')
    nltk.download('jeita')

    knbc = nltk.corpus.knbc.sents()
    jeita = nltk.corpus.jeita.sents()

    return setup_corpus(knbc), setup_corpus(jeita)

def setup_corpus(corpus):
    tagged_corpus = []

    for sentence in corpus:
        tagged_sentence = {'characters': [], 'labels': []}

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

def generate_character_type_embeddings():
    character_type_embeddings = defaultdict(int)

    character_type_embeddings['hiragana'] = 0
    character_type_embeddings['katakana'] = 1
    character_type_embeddings['kanji'] = 2
    character_type_embeddings['romaji'] = 3
    character_type_embeddings['digit'] = 4
    character_type_embeddings['other'] = 5

    bigrams = list(itertools.product(character_type_embeddings, repeat=2))
    trigrams = list(itertools.product(character_type_embeddings, repeat=3))

    for bigram in bigrams:
        character_type_embeddings[bigram[0] + bigram[1]] = len(character_type_embeddings)
    for trigram in trigrams:
        character_type_embeddings[trigram[0] + trigram[1] + trigram[2]] = len(character_type_embeddings)

    print(len(character_type_embeddings), " different character type combinations")

def get_character_type(char):
    if re.match(u'[ぁ-ん]', char):
        return 'hiragana'
    elif re.match(u'[ァ-ン]', char):
        return 'katakana'
    elif re.match(u'[一-龥]', char):
        return 'kanji'
    elif re.match(u'[A-Za-z]', char):
        return 'romaji'
    elif re.match(u'[0-9０-９]', char):
        return 'digit'
    else:
        return 'other'

def load_ngram_embeddings(embeddings_path):
    ngrams = []
    idx = 0
    ngram2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='ngram_embeddings/200D.dat', mode='w')

    with open(embeddings_path, 'rb') as f:
        next(f)  # Skip the matrix size indicator
        for l in f:
            line = l.decode().split()

            # For some reason the embeddings file contains malformed lines sometimes
            if len(line) != 200 + 1:
                continue

            word = line[0]
            ngrams.append(word)
            ngram2idx[word] = idx

            idx += 1
            vect = np.array(line[1:]).astype(float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((2000000, 200)), rootdir='ngram_embeddings/200D.dat', mode='w')
    vectors.flush()
    pickle.dump(ngrams, open('200D.pkl', 'wb'))
    pickle.dump(ngram2idx, open('200D_idx.pkl', 'wb'))

    vectors = bcolz.open('ngram_embeddings/200D.dat')[:]
    ngrams = pickle.load(open('200D.pkl', 'rb'))
    ngram2idx = pickle.load(open('200D_idx.pkl', 'rb'))

    return {n: vectors[ngram2idx[n]] for n in ngrams}

class NGram:
  def __init__(self, n, characters, character_types):
    self.n = n
    self.characters = characters
    self.character_types = character_types

  # def __str__(self):
  #  return f"{self.characters}.{list(character_type_embeddings.keys())[list(character_type_embeddings.values()).index(self.character_types)]}"

  def get_embedding(self, dimensionality, ngram_embeddings, character_type_embeddings):
    ngram_str = ''.join(self.characters)

    ngram_embedding = torch.zeros(dimensionality)
    if ngram_str in ngram_embeddings:
      ngram_embedding = np.array([Decimal(x) for x in ngram_embeddings[ngram_str]])

    character_type_embedding = torch.zeros(len(character_type_embeddings))
    character_type_embedding[self.character_types] = 1

    return np.concatenate((ngram_embedding, character_type_embedding))

class CharacterVector:
  def __init__(self, unigram, bigram, trigram):
    self.unigram = unigram
    self.bigram = bigram
    self.trigram = trigram

  # def __str__(self):
  #  return f"Unigram: {self.unigram} | Bigram: {self.bigram} | Trigram: {self.trigram}"

  def get_embeddings(self):
    return np.concatenate((self.unigram.get_embedding(), self.bigram.get_embedding(), self.trigram.get_embedding()))

def generate_character_vectors(sentence, character_type_embeddings, window_size):
  vector_count = len(sentence)

  character_vector = np.empty(shape=vector_count, dtype=CharacterVector)
  for t in range(0, vector_count):
    characters = sentence[t:t+window_size]

    character_1 = characters[0] if len(characters) > 0 else '<pad>'
    character_2 = characters[1] if len(characters) > 1 else '<pad>'
    character_3 = characters[2] if len(characters) > 2 else '<pad>'

    character_type_1 = get_character_type(character_1)
    character_type_2 = get_character_type(character_2)
    character_type_3 = get_character_type(character_3)

    unigram = NGram(1, [character_1], character_type_embeddings[character_type_1])
    bigram = NGram(2, [character_1, character_2], character_type_embeddings[character_type_1 + character_type_2])
    trigram = NGram(3, [character_1, character_2, character_3], character_type_embeddings[character_type_1 + character_type_2 + character_type_3])

    character_vector[t] = CharacterVector(unigram, bigram, trigram)

  return character_vector

def generate_label_vectors(labels):
  label_vectors = []
  label_list = [PAD_TOKEN, 'S', 'B', 'E', 'I']

  for label in labels:
    label_vector = np.zeros(len(label_list))
    label_vector[label_list.index(label)] = 1
    label_vectors.append(label_vector)

  return label_vectors

def create_loaders(device, batch_size):
    def collate_fn(batch):
        max_sentence_length = max([len(sentence['characters']) for sentence in batch])
        padded_characters = []
        padded_labels = []

        for sentence in batch:
            characters = sentence['characters']
            labels = sentence['labels']

            padded_sentence = characters + [PAD_TOKEN] * (max_sentence_length - len(characters))
            padded_character_vectors = generate_character_vectors(padded_sentence)

            character_vector_embeddings = []
            for character_vector in padded_character_vectors:
                character_vector_embeddings.append(character_vector.get_embeddings())

            padded_characters.append(character_vector_embeddings)

            padded_sentence_labels = labels + [PAD_TOKEN] * (max_sentence_length - len(labels))
            padded_labels.append(generate_label_vectors(padded_sentence_labels))

        return (torch.tensor(padded_characters, dtype=torch.float32).to(device),
                torch.tensor(padded_labels,dtype=torch.float32).to(device))


    train_data, valid_data, test_data = random_split(setup_corpora()[0], [0.8, 0.1, 0.1])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, validate_loader, test_loader

def preprocess():
    if _SET_SEED:
        random.seed(_SEED)
        np.random.seed(_SEED)
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed(_SEED)
        torch.backends.cudnn.deterministic = True

    loaders = create_loaders("cpu", 32)