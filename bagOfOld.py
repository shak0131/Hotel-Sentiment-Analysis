import math
import numpy as np

class BagOfWordsModel:
    def __init__(self, text_data):

        self.vocab = []
        self.idf_vector = []
        self.documents = len(text_data)
        word_count = {}

        for text in text_data:
            words = set(text.split())  
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1

        self.vocab = sorted(word_count.keys())
        self.idf_vector = [math.log2(self.documents / word_count[word]) for word in self.vocab]

    def transform(self, text_data):
      
        feature_matrix = []

        for text in text_data:
            word_count = {}
            words = text.split()

            for word in words:
                if word in self.vocab:
                    word_count[word] = word_count.get(word, 0) + 1
           

            tf_idf_vector = [
                (word_count.get(word, 0) / len(words)) * self.idf_vector[i]
                for i, word in enumerate(self.vocab)
            ]
            feature_matrix.append(tf_idf_vector)

        return np.array(feature_matrix)