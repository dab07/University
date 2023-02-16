import nltk
nltk.download('punkt')
nltk.download('treebank')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize

sentence = "Books are on the table"
print(word_tokenize(sentence))

sentence = "Books are on the table."
sentence = sentence.lower()
print(sentence)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
sentence = "I am Giyu Tomiako. Water Hashira in Demon Slayer corps"
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(sentence)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
print(filtered_sentence)

import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()

sentence = "Machine Learning is cool"

for word in sentence.split():
  print(ps.stem(word))

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
text = 'maching is caring'
print(lemmatizer.lemmatize("Machine", pos='n'))
print(lemmatizer.lemmatize("caring", pos='v'))
print(lemmatizer.lemmatize(text))