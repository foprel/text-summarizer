from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import SnowballStemmer
import re
import numpy as np
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class Smmry:

    def __init__(self, corpus, lang='english'):
        self.corpus = corpus
        self.lang = lang
        self.corpus_length = len(corpus)

    def sent_tokenizer(self):
        txt = self.corpus
        txt_list = []
        for line in txt.splitlines():
            line = re.sub(r"\.'|\.’|\.\"", "'.", line)
            if line and not re.search(r"\.$", line):
                line += "."
            txt_list.append(line)
        txt_string = " ".join(txt_list)
        tokens = sent_tokenize(txt_string)
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
        tokens = self.sent_tokenizer()
        sw = stopwords.words(self.lang)
        preprocessed_tokens = []
        for index, s in enumerate(tokens):
            preprocessed_tokens.append([])
            terms = word_tokenize(s)
            for t in terms:
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

    def build_similarity_matrix(self):
        tokens = self.stem()
        token_strings = [" ".join(sentence) for sentence in tokens]
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.50)
        X = vectorizer.fit_transform(token_strings)
        cosine_similarities = linear_kernel(X, X)
        for index1 in range(len(cosine_similarities)):
            for index2 in range(len(cosine_similarities)):
                if index1 == index2:
                    cosine_similarities[index1][index2] = 0
        for index in range(len(cosine_similarities)):
            if cosine_similarities[index].sum() == 0:
                continue
            else:
                cosine_similarities[index] /= cosine_similarities[index].sum()
        return cosine_similarities

    def summarize(self, length=5):
        sentence_ranks = self.textrank(self.build_similarity_matrix())
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:length])
        summary = itemgetter(*selected_sentences)(self.sent_tokenizer())
        str_summary = "\n\n".join(summary)
        return str_summary

