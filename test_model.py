from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pickle

X, y = pickle.load(open('data/data_featured.pickle', 'rb'))
X = X[:,1].reshape(-1, 1)
y = y.reshape(y.shape[0]).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
print("Apprentissage terminé")
pred = knn.predict(X_test)
print("RMSE : "+str(mean_squared_error(y_test, pred)))
print("Accuracy : "+str(accuracy_score(y_test, pred)))
