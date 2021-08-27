import pandas as pd
from nltk import word_tokenize
# file = open(file="positive-translated.csv", mode="r", encoding='utf-8')

file = pd.read_csv("tweets/positive-translated.csv", skipinitialspace=True)
print(file.columns)
print(file.index)

for twee in file['positive']:
    print(word_tokenize(twee))

