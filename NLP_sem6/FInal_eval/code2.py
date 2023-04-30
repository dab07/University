import nltk
textfile = open('/Users/hs/PycharmProjects/TwitterProject/NLP_sem6/Monkey_D_Luffy.txt','r').readlines()
textfile = ' '.join(textfile).lower()
print("________________________TOKENIZING SENTENCE________________________")
from nltk.tokenize import word_tokenize
print(word_tokenize(textfile))

print("________________________STOPWORD REMOVAL________________________")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(textfile)
no_stop_word = [w for w in word_tokens if not w in stop_words]
print(no_stop_word)

print("\n________________________STEMMING________________________")
from nltk.stem import PorterStemmer
ps = PorterStemmer()
print(ps.stem(textfile))

print("\n________________________LEMMETIZATION________________________")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize(textfile))

