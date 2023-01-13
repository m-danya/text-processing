import json
import pickle

import scipy
import sklearn

import numpy as np

DEV_DATASET_PATH = 'dev-dataset-task2022-04.json'
MODEL_SAVE_PATH = 'model.pkl'
TOKENIZER_SAVE_PATH = 'tokenizer.pkl'


class Solution:
    def __init__(self):
        with open(DEV_DATASET_PATH) as f:
            dev_dataset = json.load(f)
            self.train_x = np.array([text for (text, label) in dev_dataset])
            self.train_y = np.array([int(label) for (text, label) in dev_dataset])

        with open(MODEL_SAVE_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(TOKENIZER_SAVE_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.train_x = self.tokenizer.transform(self.train_x)

    def predict(self, text: str) -> str:
        text_v = self.tokenizer.transform([text])
        pred = self.model.predict(text_v)
        self.train_x = scipy.sparse.vstack([self.train_x, text_v])
        self.train_y = np.concatenate([self.train_y, pred])
        self.model.fit(self.train_x, self.train_y)
        return str(int(pred[0]))

