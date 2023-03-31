import nltk
import re

textfile = open('/Users/hs/PycharmProjects/TwitterProject/NLP_sem6/Monkey_D_Luffy.txt', 'r').readlines()
print("TextFile: \n", textfile)
textfile = ' '.join(textfile).lower()

from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(textfile)
# print(sentences)

# STEP 1 : Data Cleaning
dict = {}
text=""
for a in sentences:
    temp = re.sub("[^a-zA-Z]", " ", a)
    temp = temp.lower()
    dict[temp] = a
    text += temp

# Getting tf-idf score of sentences
stopwords = nltk.corpus.stopwords.words('english')
word_frequencies = {}
for word in nltk.word_tokenize(text):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

max_freq = max(word_frequencies.values())

for w in word_frequencies:
    word_frequencies[w] /= max_freq

sentence_scores = {}
for sent in sentences:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

# Summary Generation
import heapq
summary_sentences = heapq.nlargest(17, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)
print("\nSummary:\n", summary)
