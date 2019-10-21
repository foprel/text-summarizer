from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class Smmry():

    def __init__(self):
        corpus = [
            ['Two', 'wrongs', 'don\'t', 'make', 'a', 'right', '.'],
            ['The', 'pen', 'is', 'mightier', 'than', 'the', 'sword'],
            ['Don\'t', 'put', 'all', 'your', 'eggs', 'in', 'one', 'basket', '.']
        ]

    def improved_cosine(self):
        tokens = self.corpus
        def dummy(tokens):
            return tokens
        tfidf = TfidfVectorizer(analyzer='word', tokensizer=dummy, preprocessor=dummy, tokens_pattern=None)
        tfidf.fit(tokens)
        sparse_matrix = tfidf.transform(tokens)
        tokens_term_matrix = sparse_matrix.todense()
        indexes = []
        for s in range(len(tokens)):
            indexes.append(s)
        df = pd.DataFrame(tokens_term_matrix,
                          columns=tfidf.get_feature_names(),
                          index=indexes)
        return cosine_similarity(df, df)


