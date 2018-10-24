import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pickle

some_text = "Bonjour, le délai d'acheminement de vos bien est, sur la base des camions! Pourriez vous me contacter, merci."
X, y = pickle.load(open('data/data_set.pickle', 'rb'))
# print(len(nltk.word_tokenize(some_text)))

# Ajout de la taille des essais normalisés entre 0 et 1
X = X[:,None]
size = []
for x in X :
    size.append([len(x[0])])
size = np.array(size)
size = size - size.min()
size = size / size.max()
X = np.concatenate((X, size), axis=1)

pickle.dump((X,y), open("data/data_featured.pickle", "wb"))
