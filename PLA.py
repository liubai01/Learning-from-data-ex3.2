#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : PLA.py
# @Author: Yin-tao Xu
# @Date  : 18-9-16
# @Desc  : Implementation of the PLA algorithm(pocket)
import copy
import numpy as np


def eval_Err(w, X, y):
    """
    evaluate the binary error with respect to designed matrix X, label y and
    weights(contain bias) w.
    :param w: a 2-d numpy array. w.shape = (3, 1) contains weights and bias
    :param X: a 2-d numpy array. X.shape = (n, 2,)
    :param y: a 1-d numpy array. y.shape = (n,)
    :return: an integer, representing error
    """
    n = X.shape[0]
    X_aug = np.concatenate((np.ones((n, 1)), X), axis=1)
    pred = np.matmul(X_aug, w)
    pred = np.sign(pred) - (pred == 0)
    pred = pred.reshape(-1)
    return np.count_nonzero(pred == y) / n

def pocket(X_train, y_train, X_test, y_test, max_update=100):
    """
    The difference of pocket version of PLA with respect to the original
     one is that after each update, we pick the optimal weights and bias
     by picking weights with smaller Ein.
     Note: You are supposed to guratee that input data is not linear
     sperable. Early stopping with error diminishing to zero will
     raise Exception.
    :param X_train: a 2-d numpy array. X_train.shape = (n_train, 2,)
    :param y_train: a 1-d numpy array. y_train.shape = (n_train,)
    :param X_test: a 2-d numpy array. X_test.shape = (n_test, 2,)
    :param y_test: a 1-d numpy array. y_test.shape = (n_test,)
    :param max_update: an integer, representing the max iteration of updating in
    PLA algorithm
    :return: w_hat, errs_In_hat(t), errs_In(t), errs_Out_hat(t), errs_Out(t)
        w_hat: a 2-d numpy array. w.shape = (3, 1) contains weights and bias
        errs　are all 1-d numpy array. errs.shape = (max_update,), recording
        the error variation w.r.t training iteraion.
    """
    n_train = X_train.shape[0]

    X_aug = np.concatenate((np.ones((n_train, 1)), X_train), axis=1)

    w = np.zeros((X_aug.shape[1], 1))
    w_hat = np.zeros((X_aug.shape[1], 1))
    min_err = 1
    errs_train_hat = []
    errs_test_hat = []
    errs_train_now = []
    errs_test_now = []
    for u in range(max_update):
        pred = np.matmul(X_aug, w)
        pred = np.sign(pred) - (pred == 0)
        pred = pred.reshape(-1)

        i = np.random.choice(np.where(pred != y_train)[0], 1)
        w += y_train[i] * X_aug[i, :].reshape(-1, 1)
        now_err = eval_Err(w, X_train, y_train)
        # if now in sample error is less than the optimal one,
        # replace the optimal one with current one
        if now_err < min_err:
            min_err = now_err
            w_hat = copy.deepcopy(w)
        errs_train_hat.append(min_err)
        errs_train_now.append(now_err)
        errs_test_hat.append(eval_Err(w_hat, X_test, y_test))
        errs_test_now.append(eval_Err(w, X_test, y_test))

    return w_hat, \
           np.array(errs_train_hat), \
           np.array(errs_train_now), \
           np.array(errs_test_hat), \
           np.array(errs_test_now)

if __name__ == "__main__":
    from utils.data_generator import generate_samples
    from utils.visualize import visualize_data_dist
    X, y, _, _ = generate_samples(100)
    visualize_data_dist(X, y)
    pocket(X, y)
