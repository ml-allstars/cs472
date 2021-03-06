import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.pipeline import Pipeline

from mlp import mlp_driver
from utils.metrics import precision_recall_score


def display_results(score, actual, predictions):
    print('Predicted class distribution: ', Counter(predictions))
    precision, recall, other_score = precision_recall_score(actual, predictions)
    print('Score: {}'.format(score))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall), '\n')


def fit_predict(MClass, X, y):
    MClass.fit(X, y)
    return MClass.predict(X)


def fit_predict_display(MClass, X, y):
    pred = fit_predict(MClass, X, y)
    display_results(MClass.score(X, y), y, pred)


data = np.genfromtxt('data/data.csv', delimiter=',', dtype=float)

X = data[:, 0:-1]
y = data[:, -1]

print(X.shape)

# print('Initial class distribution:', Counter(y))
#
# MClass = MLPClassifier()
# MClass.fit(X, y)
# pred = MClass.predict(X)
# display_results(MClass.score(X, y), y, pred)
#
# print('SMOTE...')
# oversample = SMOTE(k_neighbors=5)
# new_X, new_y = oversample.fit_resample(X, y)
#
# print('Oversampled class distribution:', Counter(new_y))
#
# MClass.fit(new_X, new_y)
# new_pred = MClass.predict(X)
# display_results(MClass.score(X, y), y, new_pred)
#
# print('Under-sampling...')
# undersample = RandomUnderSampler()
# new_X, new_y = undersample.fit_resample(X, y)
# print('Undersampled class distribution:', Counter(new_y))
# MClass.fit(new_X, new_y)
# new_pred = MClass.predict(X)
# display_results(MClass.score(X, y), y, new_pred)
#
# print('SMOTE and under-sampling...')
# steps = [('o', oversample), ('u', undersample)]
# combo = Pipeline(steps=steps)
#
# new_X, new_y = combo.fit_resample(X, y)
# print('Combo class distribution:', Counter(new_y))
# MClass.fit(new_X, new_y)
# new_pred = MClass.predict(X)
# display_results(MClass.score(X, y), y, new_pred)


mlp_driver.run(X, y)
