import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

textfile = open('Monkey_D_Luffy.txt', 'r').readlines()
print("TextFile: \n", textfile)
textfile = ' '.join(textfile).lower()

tokenized = sent_tokenize(textfile)
for i in tokenized:
    # Word tokenizers is used to find the words
    # and punctuation in a string
    wordsList = nltk.word_tokenize(i)

    # removing stop words from wordList
    wordsList = [w for w in wordsList if not w in stop_words]

    #  Using a Tagger. Which is part-of-speech
    # tagger or POS-tagger.
    tagged = nltk.pos_tag(wordsList)

    print(tagged)