from dataclasses import dataclass
from enum import Enum, IntEnum

import torch
from torch.utils.data import DataLoader

type Corpus = list[dict[str, list[str]]]
type NGramEmbeddings = dict[str, list[float]]
type TrainValidateTest = tuple[DataLoader, DataLoader, DataLoader]

class CorpusOrigin(Enum):
    JEITA = 0,
    KNBC = 1,
    OTHER = 2,

class CharacterType(IntEnum):
    HIRAGANA = 0,
    KATAKANA = 1,
    KANJI = 2,
    ROMAJI = 3,
    DIGIT = 4,
    OTHER = 5

@dataclass
class Config:
    set_seed: bool
    seed: int
    embeddings_path: str
    custom_dataset_path: str
    window_size: int

    embedding_dim: int
    input_dim: int
    hidden_dim: int
    output_dim = 5
    layers: int
    bidirectional: bool
    dropout: float
    learning_rate: float
    epochs: int
    batch_size: int

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")