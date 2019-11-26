# smmry
A Python text summarizer containing the following functions
- sent_tokenizer: splits text into words per sentence
- preprocess: removes punctuation and removes stopwords
- tag_pos: part-of-speech tagger that selects adjectves (ADJ) and nouns(NN)
- stem: stems words in sentences
- text_rank: ranks sentences based on text rank algorithm
- build_similarity_matrix: creates a cosine similarity matrix based on tfidf vectors
- summarize: runs through all the steps above and generates a summary
