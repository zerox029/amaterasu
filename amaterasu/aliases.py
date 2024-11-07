from enum import Enum, IntEnum

type Corpus = list[dict[str, list[str]]]
type NGramEmbeddings = dict[str, list[float]]

class CharacterType(IntEnum):
    HIRAGANA = 0,
    KATAKANA = 1,
    KANJI = 2,
    ROMAJI = 3,
    DIGIT = 4,
    OTHER = 5
