# from warnings import warn
import numpy as np
from data_utils import *
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

data = gather_and_clean_data()

X = data[:, 0:-1]
y = data[:, -1]

MClass = MLPClassifier()
MClass.fit(X, y)
pred = MClass.predict(X)
score = MClass.score(X, y)
print(f"Pred: {pred}")
print(f"Score: {score}")
