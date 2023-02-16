import matplotlib.pyplot as plt
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

textfile = open('/Users/hs/PycharmProjects/TwitterProject/NLP_sem6/Monkey_D_Luffy.txt','r').readlines()
print(textfile)
textfile = ' '.join(textfile).lower()

# FREQUENCY
wordcloud = WordCloud(stopwords = STOPWORDS,
                      collocations=True).generate(textfile)


myDict = wordcloud.process_text(textfile)
word_freq = {k: v for k, v in sorted(myDict.items(),reverse=True, key=lambda item: item[1])}
print("_________________Frequency of Words in the text file in sorted order_________________")
a = 0
b = 0
length = len(myDict) / 7
for i in range(1, int(length)):
    b = b + 7
    print(list(word_freq.items())[a:b])
    a = b + 1

# Displaying Graph os word Cloud
# plt.imshow(wordcloud, interpolation='bilInear')
plt.hist(myDict)
plt.axis('on')
plt.show()


