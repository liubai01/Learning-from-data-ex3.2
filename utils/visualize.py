#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : visualize.py
# @Author: Yin-tao Xu
# @Date  : 18-9-16
# @Desc  : visualize the generated data
import matplotlib.pyplot as plt

def visualize_data_dist(X, y):
    """
    Visualize the datset distribution
    :param X: a 2-d numpy array. X.shape = (n, 2,)
    :param y: a 1-d numpy array. y.shape = (n,)
    :return: None
    """
    # dump postive and negative data points into lists
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []
    for i in range(X.shape[0]):
        if y[i] == 1:
            pos_x.append(X[i, 0])
            pos_y.append(X[i, 1])
        else:
            neg_x.append(X[i, 0])
            neg_y.append(X[i, 1])
    # visualization
    plt.scatter(pos_x, pos_y, c='', edgecolors='b', marker='o', s=50)
    plt.scatter(neg_x, neg_y, c='r', marker='x', s=50)
    plt.show()

if __name__ == "__main__":
    from utils.data_generator import generate_training_samples
    X, y, _, _ = generate_training_samples(100)
    visualize_data_dist(X, y)
