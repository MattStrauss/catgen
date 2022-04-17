import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin


# https://lvngd.com/blog/spacy-word-vectors-as-features-in-scikit-learn/
# transforms the csv of titles into 300 dimension word vectors
# using the Spacy "en_core_web_lg" model
class WordVectorTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, model="en_core_web_lg"):
        self.model = model

    # the unused params are needed to extend the proper
    # def of the base class
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        nlp = spacy.load(self.model)
        return np.concatenate([nlp(doc).vector.reshape((1, -1)) for doc in x])
