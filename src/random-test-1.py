from nltk.util import ngrams

sentence = "I am learning about n-grams"
n_grams = list(ngrams(sentence.split(), 2))  # 2-grams (bigrams)
print(n_grams)
