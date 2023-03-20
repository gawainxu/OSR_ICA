import numpy as np
import pickle
import random

import torch
from torch.utils.data import Dataset

EPS = 1e-4

def sample_integers(n, shape):
    sample = np.random.choice(n, size=shape, replace=True)
    sample = (np.rint(sample)).astype(int)
    return sample


def np_scatter(a, row_idx):
    #print(a.shape)
    new_f = np.empty([a.shape[-1]])
    for i, r in enumerate(row_idx):
        elem = a[r, i]
        new_f[i] = elem

    return new_f


def resample_rows_per_column(x):
    # x is numpy array
    n_batch = x.shape[0]
    n_dim = x.shape[1]
    x_permutes = []
    for i in range(n_batch):
        row_indices = sample_integers(n_batch, n_dim)    # sample with repeating
        x_permute = np_scatter(x, row_indices)
        x_permutes.append(x_permute)
    x_permutes = np.array(x_permutes)

    return x_permutes


class ica_dataset(Dataset):
    def __init__(self, feature_path, feature_name, train_test_ratio=0, if_train=0):
        super().__init__()

        self.if_train = if_train

        with open(feature_path, "rb") as f:
            features, labels = pickle.load(f)
        features = [feature[feature_name].detach().numpy() for feature in features]
        num_features = len(features)
        self.features = np.squeeze(np.array(features))                                    # n_batch * n_dim
        #print(self.features.shape)
        if self.features.ndim > 2:
            self.features = np.reshape(self.features, (num_features, self.features.shape[1]*self.features.shape[2]))
        #print(self.features.shape)

        # normalization, TODO if it is right?
        features_mean = np.mean(self.features, axis=0)
        features_sd = np.sqrt(np.mean(np.power(self.features, 2), axis=0)+EPS)
        self.features = self.features - features_mean
        self.features = self.features / features_sd

        # split training and testing, TODO no need to split, yes?
        #train_indices = list(range(0, num_features, int(1./(1-train_test_ratio))))
        #testing_indices = [i for i in range(num_features) if i not in train_indices]

        #self.train_features = self.features[train_indices]
        #self.testing_features = self.features[testing_indices]

        #self.train_features_permute = resample_rows_per_column(self.train_features)
        #self.testing_features_permute = resample_rows_per_column(self.testing_features)
        # TODO this should be done in mini-batch
        #self.features_permute = resample_rows_per_column(self.features)

    def __len__(self):
        #if self.if_train == True:
        #    return len(self.train_features)
        #else:
        #    return len(self.testing_features)
        return len(self.features)

    def __getitem__(self, index):

        #if self.if_train == "Train":
        #    return self.train_features[index], self.train_features_permute[index]
        #else:
        #    return self.testing_features[index], self.testing_features_permute[index]
        return self.features[index]


if __name__ == "__main__":

    featurePath = "D://projects//open_cross_entropy//osr_closed_set_all_you_need-main//features//cifar-10-10_classifier32_0"
    feature_name = "module.avgpool"

    dataset = ica_dataset(feature_path=featurePath, feature_name=feature_name)
    dj = dataset[0:20]
    #dj = np.expand_dims(dj, 0)
    dm = resample_rows_per_column(dj)
    print(dm.shape)