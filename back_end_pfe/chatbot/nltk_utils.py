# import numpy as np
# import nltk
# # nltk.download('punkt')
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()


# def tokenize(sentence):
#     return nltk.word_tokenize(sentence)


# def stem(word):

#     return stemmer.stem(word.lower())


# def bag_of_words(tokenized_sentence, words):
#     # stem each word
#     sentence_words = [stem(word) for word in tokenized_sentence]
#     # initialize bag with 0 for each word
#     bag = np.zeros(len(words), dtype=np.float32)
#     for idx, w in enumerate(words):
#         if w in sentence_words: 
#             bag[idx] = 1

#     return bag

import re
from nltk.corpus import stopwords
from string import punctuation

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove stopwords
    stop_words = set(stopwords.words('fransh'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text




# import nltk
# import numpy as np
# from nltk.stem.porter import PorterStemmer


# # nltk_utils.py
# def pad_sequence(sequences, max_length):
#     padded_sequences = []
#     for seq in sequences:
#         if len(seq) < max_length:
#             padded_seq = seq + [0] * (max_length - len(seq))
#         else:
#             padded_seq = seq[:max_length]
#         padded_sequences.append(padded_seq)
#     return padded_sequences

