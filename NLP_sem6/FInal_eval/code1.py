import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud
from textblob import TextBlob
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models

df = pd.read_csv("/Users/hs/UNI_Material/Datasets/IMDB Dataset.csv")
# print(df)
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
df['review']= df['review'].apply(lambda x:remove_punctuation(x))
df['review']= df['review'].apply(lambda x: x.lower())

df['Review']=df['review'].astype(str)
def tokenize_review(review):
    tokens = word_tokenize(review)
    return tokens
df['tokens'] = df['review'].apply(tokenize_review)
stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output
df['tokens']= df['tokens'].apply(lambda x:remove_stopwords(x))

freq = {}
for i in df['lemmatize']:
    for j in i:
        if j not in freq:
            freq[j] = 1
        else:
            freq[j] += 1
list(freq.items())[:25]
freq_df=pd.DataFrame(sorted(freq.items(),key=lambda x:x[1],reverse=True))
f1=freq_df[0][:10]
f2=freq_df[1][:10]
plt.figure(1,figsize=(16,4))
plt.bar(f1,f2,color ='blue',width = 0.4)
plt.xlabel("Words in the dataframe")
plt.ylabel("Count")
plt.title("Top 10 words in the dataframe")
plt.savefig("wordfrequency.png")
plt.show()

text = ' '.join(df['review'])
wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("wordcloud.png")
plt.show()
