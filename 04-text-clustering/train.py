# копировал из ноутбука, поэтому кодстайл такой странный

import json

DEV_DATASET_PATH = 'dev-dataset-task2022-04.json'
MODEL_SAVE_PATH = 'model.pkl'
TOKENIZER_SAVE_PATH = 'tokenizer.pkl'

with open(DEV_DATASET_PATH) as f:
    dev_dataset = json.load(f)

import nltk
import numpy as np
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
tf_idf = TfidfVectorizer(min_df=3, stop_words=stopwords.words('russian'))


train_x = np.array([text for (text, label) in dev_dataset])
# print(train_x)
train_x = tf_idf.fit_transform(train_x)

train_y = np.array([int(label) for (text, label) in dev_dataset])

import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer)
       and returns num_folds train-val pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list
                                       contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without
                                       i-th fold while the 2nd one contains
                                       i-th fold
    """
    k = num_objects // num_folds
    all = list(range(num_objects))
    fold = [all[k * x: k * (x + 1) if x < num_folds - 1 else num_objects] for x
            in range(num_folds)]
    other = [[x for x in all if x not in f] for f in fold]
    return list(zip(other, fold))


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    cross_val_score_dict = {}
    for n_neighbors in parameters['n_neighbors']:
        for metric in parameters['metrics']:
            for weight in parameters['weights']:
                for normalizer_and_name in parameters['normalizers']:
                    folds_results = []
                    normalizer, normalizer_name = normalizer_and_name
                    for train_idxs, test_idxs in folds:
                        x_train = X[train_idxs]
                        model = knn_class(n_neighbors=n_neighbors,
                                          metric=metric, weights=weight)
                        if normalizer:
                            s = normalizer
                            s.fit(x_train)
                            x_train = s.transform(x_train)
                        model.fit(x_train, y[train_idxs])

                        x_test = X[test_idxs]
                        if normalizer:
                            x_test = s.transform(x_test)
                        y_test = model.predict(x_test)
                        folds_results.append(
                            score_function(y[test_idxs], y_test))
                    model_result = sum(folds_results) / len(folds_results)
                    cross_val_score_dict[(normalizer_name, n_neighbors, metric,
                                          weight)] = model_result
    return cross_val_score_dict

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

parameters = {
    'n_neighbors': [1, 2, 3, 5, 10],
    'metrics': ['euclidean', 'cosine'],
    'weights': ['uniform', 'distance'],
    'normalizers': [(None, 'None'), (MinMaxScaler(), 'MinMax'), (StandardScaler(), 'Standard')]
}


# folds = kfold_split(len(train_x), 5)
# cross_val_results = knn_cv_score(x_train, y_train, parameters, accuracy_score, folds, neighbors.KNeighborsRegressor)

# best_params = max(cross_val_results, key=cross_val_results.get)
# print(best_params, 'gives score ', max(cross_val_results.values()))

from sklearn import neighbors

model = neighbors.KNeighborsRegressor(n_neighbors=1, metric='cosine')
model.fit(train_x, train_y)
import pickle

with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)

with open(TOKENIZER_SAVE_PATH, 'wb') as f:
    pickle.dump(tf_idf, f)