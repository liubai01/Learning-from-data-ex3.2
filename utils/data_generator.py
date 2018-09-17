#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_generator.py
# @Author: Yin-tao Xu
# @Date  : 18-9-16
# @Desc  : a tiny tool to generate 2-d data pts. as metioned in
# LFD (Learning from data) p80 Exercise 3.2.

import random

import numpy as np


def generate_samples(n_train=100, n_test=1000):
    """
    generate 2-d data pts, of which
    x_1 \in [-1, 1], x_2 \in [-1, 1]. First, choose labels by
    a random line. Flip N / 10 randomly selected y_n' s. Resample
    if the dataset is too imbalanced
    :param n: the required quantity of the trainning sample
    :return: X_train, y_train, X_test, y_test
        X_train: a 2-d numpy array. X_train.shape = (n_train, 2,)
        y_train: a 1-d numpy array. y_train.shape = (n_train,)
        X_test: a 2-d numpy array. X_test.shape = (n_test, 2,)
        y_test: a 1-d numpy array. y_test.shape = (n_test,)
    """

    # initialize the arraies
    X_train = np.zeros((n_train, 2), dtype=np.float)
    X_test = np.zeros((n_test, 2), dtype=np.int)
    y_train = np.zeros((n_train, ), dtype=np.float)
    y_test = np.zeros((n_test, ), dtype=np.int)

    # x_1 \in [-1, 1], (for numerical issue, the pratical
    # range is [-1, 1) )
    X_train[:, 0] = np.random.rand(n_train) * 2 - 1
    X_test[:, 0] = np.random.rand(n_test) * 2 - 1
    # x_2 \in [-1, 1], (for numerical issue, the pratical
    # range is [-1, 1) )
    X_train[:, 1] = np.random.rand(n_train) * 2 - 1
    X_test[:, 1] = np.random.rand(n_test) * 2 - 1

    while True:
        # generate a random line
        w_1 = random.random() / 0.4 - 0.2
        w_2 = random.random() / 0.4 - 0.2
        b = random.random() / 0.4 - 0.2

        for i in range(n_train):
            y_train[i] = np.sign(w_1 * X_train[i, 0] + w_2 * X_train[i, 1] + b)
        # if the dataset is imblanced, resample
        if np.abs((np.count_nonzero(y_train == 1) + 1) /
                          (np.count_nonzero(y_train == -1) + 1) - 0.5) < 0.1:
            break
    # flip the labels of the 10% dataset
    flip_y = np.random.choice(n_train, n_train // 10, replace=True)
    y_train[flip_y] = -y_train[flip_y]

    for i in range(n_test):
        y_test[i] = np.sign(w_1 * X_test[i, 0] + w_2 * X_test[i, 1] + b)
    flip_y = np.random.choice(n_test, n_test // 10, replace=True)
    y_test[flip_y] = -y_test[flip_y]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    from utils.visualize import visualize_data_dist
    X_train, y_train, _, _ = generate_samples(100)
    visualize_data_dist(X_train, y_train)


