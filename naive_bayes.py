import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
    
        self.priors = {}
        self.likelihoods = {}

    def fit(self, X, y):

    
        classes = np.unique(y)
        self.priors = {cls: np.mean(y == cls) for cls in classes}
        

        self.likelihoods = {cls: X[y == cls].mean(axis=0) for cls in classes}

    def predict(self, X):

        posteriors = []
        for x in X:
            class_scores = {
                cls: np.log(self.priors[cls]) + np.sum(np.log(self.likelihoods[cls] + 1e-9) * x)
                for cls in self.priors
            }
            posteriors.append(max(class_scores, key=class_scores.get))
        return posteriors