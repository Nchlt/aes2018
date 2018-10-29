from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter
import numpy as np
import pickle
import nltk

def lexical_diversity(text):
    return len(set(text)) / len(text)

some_text = "Bonjour, le délai d'acheminement de vos bien est, sur la base des camions! Pourriez vous me contacter, merci."
X, y = pickle.load(open('data/data_set.pickle', 'rb'))
# print(len(nltk.word_tokenize(some_text)))

# Ajout de la taille des essais normalisés entre 0 et 1
X = X[:,None]

print(nltk.word_tokenize(X[0][0]))

only_tags = []
for x in X[:,0]:
	tokenized = nltk.pos_tag(x)
	tags = list(map(itemgetter(1), tokenized))
	only_tags.append(' '.join(tags))

only_tags = np.array(only_tags)

vectorizer = TfidfVectorizer(sublinear_tf=True, norm='l2', ngram_range=(1, 3))
tf_tags = vectorizer.fit_transform(only_tags)

# tags_list = np.asarray(list(map(itemgetter(1), np.asarray(nltk.pos_tag()))))
size = []
diversity = []
for x in X :
	size.append([len(x[0])])
	diversity.append([lexical_diversity(x[0])])
size = np.array(size)
size = size - size.min()
size = size / size.max()


X = np.concatenate((size, diversity), axis=1)


print(X.shape)
print(y.shape)
pickle.dump((X,y,tf_tags), open("data/data_featured.pickle", "wb"))
