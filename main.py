#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Yin-tao Xu
# @Date  : 18-9-16
# @Desc  : The entry of this repository.

import matplotlib.pyplot as plt
import numpy as np

from PLA import pocket
from utils.data_generator import generate_samples

MAX_UPDATES = 1000
N_TRAIN_SAMPLES = 100
N_TEST_SAMPLES = 1000
N_EXP = 40

# initialize the errors

index = [i for i in range(MAX_UPDATES)]
errIn_hat_mean = np.zeros((MAX_UPDATES,), dtype=np.float)
errIn_now_mean = np.zeros((MAX_UPDATES,), dtype=np.float)
errOut_hat_mean = np.zeros((MAX_UPDATES,), dtype=np.float)
errOut_now_mean = np.zeros((MAX_UPDATES,), dtype=np.float)

# compute the mean in-sample error and out-of-sample error.

for e in range(N_EXP):

    X_train, y_train, X_test, y_test = generate_samples(N_TRAIN_SAMPLES,  N_TEST_SAMPLES)
    w_hat, errIn_hat, errIn_now, errOut_hat, errOut_now = \
        pocket(X_train, y_train, X_test, y_test, max_update=MAX_UPDATES)
    errIn_hat_mean += errIn_hat
    errIn_now_mean += errIn_now
    errOut_hat_mean += errOut_hat
    errOut_now_mean += errOut_now
    print("exp: {} | errIn_hat: {} | errOut_hat: {}".format(e + 1, errIn_hat[-1], errOut_hat[-1]))

errIn_hat_mean /= MAX_UPDATES
errOut_hat_mean /= MAX_UPDATES
errIn_now_mean /= MAX_UPDATES
errOut_now_mean /= MAX_UPDATES

# visualize the section

plt.subplot(2, 1, 1)
plt.plot(index, errIn_hat_mean, label='$E_{in}(\hat{w})$')
plt.plot(index, errIn_now_mean, label='$E_{in}(w(t))$')
plt.legend()
plt.title("$E_{in}$")

plt.subplot(2, 1, 2)
plt.plot(index, errOut_hat_mean, label='$E_{out}(\hat{w})$')
plt.plot(index, errOut_now_mean, label='$E_{out}(w(t))$')
plt.legend()
plt.title("$E_{out}$")

plt.show()
