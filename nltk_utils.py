import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Automatically download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    A token can be a word, punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Stemming = find the root form of the word.
    Examples:
        words = ["organize", "organizes", "organizing"]
        words = [stem(w) for w in words]
        -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Return a bag-of-words array:
    1 for each known word that exists in the sentence, 0 otherwise.

    Example:
        sentence = ["hello", "how", "are", "you"]
        words    = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag      = [  0,      1,    0,    1,     0,      0,       0  ]
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
