import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Auto-download required NLTK data if not present
for pkg in ('punkt', 'punkt_tab'):
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)

stemmer = PorterStemmer()


def tokenize(sentence):
    """Split sentence into array of words/tokens."""
    return nltk.word_tokenize(sentence)


def stem(word):
    """Return the root/stem form of a word."""
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Return bag-of-words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
