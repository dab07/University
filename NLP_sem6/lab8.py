import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

textfile = open('Monkey_D_Luffy.txt', 'r').readlines()
print("TextFile: \n", textfile)
textfile = ' '.join(textfile).lower()

lower_case = textfile.lower()
tokens = nltk.word_tokenize(lower_case)
tags = nltk.pos_tag(tokens)
counts = Counter( tag for word,  tag in tags)
print(counts)

fd = nltk.FreqDist(tokens)
fd.plot()

bigram_pos = list(nltk.bigrams(tokens))
print(bigram_pos)

#Tagging sentences
sentence = nltk.sent_tokenize(textfile)
for sent in sentence:
	 print(nltk.pos_tag(nltk.word_tokenize(sent)))
