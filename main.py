# from warnings import warn
import numpy as np
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

data = np.genfromtxt('data.csv', delimiter=',', dtype=float)

X = data[:, 0:-1]
y = data[:, -1]

MClass = MLPClassifier()
MClass.fit(X, y)
pred = MClass.predict(X)
score = MClass.score(X, y)
print(f"Pred: {pred}")
print(f"Score: {score}")
