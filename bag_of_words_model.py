# import math
# import numpy as np

# class BagOfWordsModel:
#     def __init__(self, text_data):

#         self.vocab = []
#         self.idf_vector = []
#         self.documents = len(text_data)
#         word_count = {}

#         for text in text_data:
#             words = set(text.split())  
#             for word in words:
#                 word_count[word] = word_count.get(word, 0) + 1

#         self.vocab = sorted(word_count.keys())
#         self.idf_vector = [math.log2(self.documents / word_count[word]) for word in self.vocab]

#     def transform(self, text_data):
      
#         feature_matrix = []

#         for text in text_data:
#             word_count = {}
#             words = text.split()

#             for word in words:
#                 if word in self.vocab:
#                     word_count[word] = word_count.get(word, 0) + 1
           

#             tf_idf_vector = [
#                 (word_count.get(word, 0) / len(words)) * self.idf_vector[i]
#                 for i, word in enumerate(self.vocab)
#             ]
#             feature_matrix.append(tf_idf_vector)

#         return np.array(feature_matrix)

import math
import numpy as np
from collections import defaultdict

class BagOfWordsModel:
    def __init__(self, text_data, n):
        self.vocab = []
        self.idf_vector = []
        self.documents = len(text_data)
        word_count = defaultdict(int)
        self.n = n

        for text in text_data:
            n_grams = self.get_n_grams(text, n)
            for n_gram in n_grams:
                word_count[n_gram] += 1

        self.vocab = sorted(word_count.keys())
        self.idf_vector = [math.log2(self.documents / word_count[n_gram]) for n_gram in self.vocab]

    def get_n_grams(self, text, n):
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)] 

    def transform(self, text_data):
        feature_matrix = []

        for text in text_data:
            word_count = defaultdict(int)
            n_grams = self.get_n_grams(text, self.n)
            if len(n_grams) <= 1:
                n_grams = self.get_n_grams(text, 1)

            for n_gram in n_grams:
                if n_gram in self.vocab:
                    word_count[n_gram] += 1
                    

            tf_idf_vector = [
                (word_count.get(n_gram, 0) / len(n_grams)) * self.idf_vector[i]
                for i, n_gram in enumerate(self.vocab)
            ]
            feature_matrix.append(tf_idf_vector)

        return np.array(feature_matrix)