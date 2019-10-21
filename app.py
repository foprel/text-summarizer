from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import SnowballStemmer
from nltk.cluster.util import cosine_distance
import re
import numpy as np
from operator import itemgetter
import time

class Smmry:

    def __init__(self, corpus, lang='english'):
        self.corpus = corpus
        self.lang = lang
        self.corpus_length = len(corpus)

    def sent_tokenize(self):
        tokens = sent_tokenize(self.corpus)
        sentences = []
        for s in tokens:
            s = re.sub("\n", " ", s)
            s = re.sub(" +", " ", s)
            s.strip()
            sentences.append(s)
        return sentences

    def tokenize(self):
        tokens = [word_tokenize(t) for t in sent_tokenize(self.corpus)]
        return tokens

    def preprocess(self):
        tokens = self.tokenize()
        sw = stopwords.words(self.lang)
        preprocessed_tokens = []
        for index, s in enumerate(tokens):
            preprocessed_tokens.append([])
            for t in s:
                t = re.sub("\n", " ", t)
                t = re.sub("[^A-Za-z]+", " ", t)
                t = re.sub(" +", " ", t)
                t = t.strip()
                t = t.lower()
                if t and t not in sw:
                    preprocessed_tokens[index].append(t)
        return preprocessed_tokens

    def tag_pos(self):
        preprocessed_tokens = self.preprocess()
        tagged_tokens = []
        for index, s in enumerate(preprocessed_tokens):
            tagged_tokens.append([])
            for t in s:
                t = pos_tag([t])
                if t[0][1] == 'NN' or 'ADJ':
                    tagged_tokens[index].append(t[0][0])
        return tagged_tokens

    def stem(self):
        tagged_tokens = self.tag_pos()
        stemmer = SnowballStemmer(self.lang)
        stemmed_tokens = []
        for index, s in enumerate(tagged_tokens):
            stemmed_tokens.append([])
            for t in s:
                t = stemmer.stem(t)
                stemmed_tokens[index].append(t)
        return stemmed_tokens

    def textrank(self, A, eps=0.0001, d=0.85):
        P = np.ones(len(A)) / len(A)
        while True:
            new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
            delta = abs(new_P - P).sum()
            if delta <= eps:
                return new_P
            P = new_P

    def cosine_similarity(self, sent1, sent2):

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        for w in sent1:
            vector1[all_words.index(w)] += 1

        for w in sent2:
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self):
        tokens = self.stem()
        S = np.zeros((len(tokens), len(tokens)))
        for idx1 in range(len(tokens)):
            for idx2 in range(len(tokens)):
                if idx1 == idx2:
                    continue
                S[idx1][idx2] = self.cosine_similarity(tokens[idx1], tokens[idx2])
        for idx in range(len(S)):
            if S[idx].sum() == 0:
                continue
            else:
                S[idx] /= S[idx].sum()
        return S

    def summarize(self, length=5):
        start_time = time.time()
        sentence_ranks = self.textrank(self.build_similarity_matrix())
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:length])
        summary = itemgetter(*selected_sentences)(self.sent_tokenize())
        str_summary = "\n\n".join(summary)
        print(str_summary)
        print(f"Time elapsed: {(time.time() - start_time)}")
        print(f"Reduced by: {100 - round((len(str_summary) / self.corpus_length * 100))}%")