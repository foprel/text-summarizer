# smmry
A Python text summarizer based containing the following functions
- tokenize: splits text into words per sentence
- preprocess: removes punctuation and removes stopwords
- tag_pos: part-of-speech tagger that selects adjectves (ADJ) and nouns(NN)
- stem: stems words in sentences
- build_similarity_matrix: creates a cosine similarity matrix
- summarize: runs through all the steps above
