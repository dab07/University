textfile1 = open('doc1.txt', 'r').readlines()
textfile1 = ''.join(textfile1).lower()

textfile2 = open('doc2.txt', 'r').readlines()
textfile2 = ' '.join(textfile2).lower()

# print(textfile1)
# print(textfile2)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
corpus = [textfile1,textfile2]

cnt = count_vect.fit_transform(corpus)
tables = pd.DataFrame(cnt.toarray(),columns=count_vect.get_feature_names(),index=['textfile 1','textfile 2'])
# print(tables)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
trsfm=vectorizer.fit_transform(corpus)
pd.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=['textfile 1','textfile 2'])

#Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
res = cosine_similarity(trsfm[0:1], trsfm)
print("Cosine Similarity", res)

textfile1 = set(textfile1.lower().split())
textfile2 = set(textfile2.lower().split())
# print(textfile1)
# print(textfile2)
intersection = textfile1.intersection(textfile2)
union = textfile1.union(textfile2)
print("Jaccard Similarity", float(len(intersection)) / len(union))

table = cnt.todense()
df = pd.DataFrame(table,
                  columns=count_vect.get_feature_names(),
                  index=['textfile 1','textfile 2'])

from scipy.spatial import distance
cnt = distance.cdist(df, df, 'euclidean')
df_eucl = pd.DataFrame(cnt,
                  columns= ['textfile 1','textfile 2'],
                  index=['textfile 1','textfile 2'])
print("Euclidean Distance:\n", df_eucl)
