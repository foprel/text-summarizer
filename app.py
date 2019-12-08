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

    def sent_tokenizer(self, txt):
        """
        Parses raw text into 2D list words per sentence using nltk.tokenize.sent_tokenize().
        Fixes issues with sentences that do not end with a dot.
        Returns: 2D List of words per sentence: [[w1, w2, w3], [w4, w5, w6], ... [wi, wi, wi]]
        @params:
            txt   -Required  : Raw text to parse (Str)
        """
        txt_list = []
        for line in txt.splitlines():
            line = re.sub(r"\.'|\.\"", "'.", line)
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

    def preprocess(self, sentences):
        """
        Preprocesses sentence tokens: 1) removes stopwords, 2) removes special characters, 3) removes and leading and
        trailing spaces, and 4) transforms all words to lowercase.
        Returns: 2D List of words per sentence: [[w1, w2, w3], [w4, w5, w6], ... [wi, wi, wi]]
        @params:
            sentences   -Required  : 2D list of words per sentence (Lst)
        """
        tokens = sentences
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

    def tag_pos(self, preprocessed_tokens, pos=['NN', 'ADJ']):
        """
        Filters only relevant parts-of-speech for further processing and/or analysis.
        Returns: 2D List of words per sentence: [[w1, w2, w3], [w4, w5, w6], ... [wi, wi, wi]]
        @params:
            preprocessed_tokens   -Required  : 2D list of preprocessed words per sentence (Lst)
            pos                   -Optional  : Parts-of-speech relevant for further processing and/or analysis.
                                               pos=['NN', 'ADJ'] by default (Lst).
        """
        tagged_tokens = []
        for index, s in enumerate(preprocessed_tokens):
            tagged_tokens.append([])
            for t in s:
                t = pos_tag([t])
                if t[0][1] in pos:
                    tagged_tokens[index].append(t[0][0])
        return tagged_tokens

    def stem(self, tagged_tokens):
        """
        Stems words using nltk.stem.SnowballStemmer.
        Returns: 2D List of words per sentence: [[w1, w2, w3], [w4, w5, w6], ... [wi, wi, wi]]
        @params:
            tagged_tokens   -Required  : 2D list of parts-of-speech tagged words per sentence (Lst)
        """
        stemmer = SnowballStemmer(self.lang)
        stemmed_tokens = []
        for index, s in enumerate(tagged_tokens):
            stemmed_tokens.append([])
            for t in s:
                t = stemmer.stem(t)
                stemmed_tokens[index].append(t)
        return stemmed_tokens

    def textrank(self, A, eps=0.0001, d=0.85):
        """
        Applies TextRank algorithm to pairwise similarity matrix.
        Returns: Ranked sentences unsorted (numpy.ndarray)
        @params:
            A       -Required  : Pairwise similarity matrix (Lst)
            eps     -Optional  : stop the algorithm when the difference between 2 consecutive iterations is smaller or
                                 equal to eps. eps=0.0001 by default (Flt)
            d       -Optional  : damping factor: With a probability of 1-d the user will simply pick a web page at random
                                 as the next destination, ignoring the link structure completely. d=0.85 by defaul (Flt)
        """
        P = np.ones(len(A)) / len(A)
        while True:
            new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
            delta = abs(new_P - P).sum()
            if delta <= eps:
                return new_P
            P = new_P

    def build_similarity_matrix(self, stemmed_tokens):
        """
        Creates tfidf vector using sklearn.feature_extraction.text.TfidfVectorizer() and builds pairwise similarity
        matrix of linear kernals using sklearn.metrics.pairwise.linear_kernel().
        Returns: Pairwise similarity matrix (numpy.ndarray)
        @params:
            stemmed_tokens   -Required  : 2D list of stemmed words per sentence (Lst)
        """
        token_strings = [" ".join(sentence) for sentence in stemmed_tokens]
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
        """
        Creates pipeline of summarization steps: 1) sent_tokenizer(), 2) preprocessing(), 3) tag_pos(), 4) stem(),
        5) build_similarity_matrix(), and 6) textrank().
        Returns: Summary (Str) in which each sentence is separated by a new line.
        @params:
            length   -Optional  : Number of sentences to be returned by function. length=5 by default (Int)
        """
        sentences = self.sent_tokenizer(self.corpus)
        preprocessed_tokens = self.preprocess(sentences)
        tagged_tokens = self.tag_pos(preprocessed_tokens)
        stemmed_tokens = self.stem(tagged_tokens)
        sentence_ranks = self.textrank(self.build_similarity_matrix(stemmed_tokens))
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:length])
        summary = itemgetter(*selected_sentences)(self.sent_tokenizer(self.corpus))
        str_summary = "\n\n".join(summary)
        return str_summary


