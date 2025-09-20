from sklearn.feature_extraction.text import CountVectorizer

class NLPModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def bow_features(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()
