# smmry
<<<<<<< HEAD
A Python text summarizer based containing the following functions
- tokenize: splits text into words per sentence
- preprocess: removes punctuation and removes stopwords
- tag_pos: part-of-speech tagger that selects adjectves (ADJ) and nouns(NN)
- stem: stems words in sentences
- build_similarity_matrix: creates a cosine similarity matrix
- summarize: runs through all the steps above
=======
A Python text summarizer containing the following functions
- sent_tokenizer: splits text into words per sentence
- preprocess: removes punctuation and removes stopwords
- tag_pos: part-of-speech tagger that selects adjectves (ADJ) and nouns(NN)
- stem: stems words in sentences
- text_rank: ranks sentences based on text rank algorithm
- build_similarity_matrix: creates a cosine similarity matrix based on tfidf vectors
- summarize: runs through all the steps above and generates a summary
>>>>>>> 6be47937625e8fd8bc2768452282696b0c1cb202
