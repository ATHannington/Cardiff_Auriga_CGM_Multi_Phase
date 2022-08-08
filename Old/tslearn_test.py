import numpy as np
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import multiprocessing as mp

import h5py
import sys
import logging
import time

n_jobs = -2
n_init = 3

pi = 3.141596


n_n_min = 10
n_n_max = 500
n_n_delta = 10

n_t = 500
n_q = 8

delta_t = 1

print_percent = 1.0


def lin_func(x, m, c):
    y = [m * val + c for val in x]
    return y


def sin_func(x, l, a):
    y = [a * np.sin(2 * pi * val / l) for val in x]
    return y


plot_x_dat = []
plot_y_dat = []
for n_n in np.arange(n_n_min, n_n_max + n_n_delta, n_n_delta):
    print("\n" + f"n_n = {n_n}")

    n_t = int(n_t * delta_t)
    n_n_tmp = int((n_n / 2))

    # print(f"n_t = {n_t}")
    # print(f"n_n_tmp = {n_n_tmp}")

    x_dat = [i for i in np.arange(0, n_t, delta_t)]

    sin_dat = np.array(
        [
            sin_func(x_dat, float(n_t) / float(l), n_t)
            for l in np.arange(1, n_n_tmp + 1, 1)
        ]
    )
    # print("Done sin_dat!")
    lin_dat = np.array([lin_func(x_dat, m, 0.0) for m in np.arange(1, n_n_tmp + 1, 1)])
    # print("Done lin_dat!")

    sin_dat_normed = []
    for tseries in sin_dat:
        maxVal = np.max(tseries)
        sin_dat_normed.append(tseries / maxVal)
    sin_dat_normed = np.array(sin_dat_normed)

    lin_dat_normed = []
    maxVal = np.max(lin_dat)
    for tseries in lin_dat:
        lin_dat_normed.append(tseries / maxVal)
    lin_dat_normed = np.array(lin_dat_normed)

    # print(f"np.shape(sin_dat) = {np.shape(sin_dat)}")
    # print(f"np.shape(lin_dat) = {np.shape(lin_dat)}")
    A = np.concatenate((sin_dat_normed, lin_dat_normed), axis=0)
    # print(f"np.shape(A) = {np.shape(A)}")
    Mtmp = [A for ii in range(0, int(n_q))]
    # B = np.concatenate((lin_dat_normed,lin_dat_normed),axis=0)
    # print(f"np.shape(B) = {np.shape(B)}")
    # Mtmp = Mtmp + [B for ii in range(0,int(n_q/2))]
    M = np.array(Mtmp).reshape(n_n, n_t, n_q)
    # print(f"np.shape(M) = {np.shape(M)}")
    del sin_dat, lin_dat

    X = to_time_series_dataset(M)

    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)

    start = time.time()
    km = TimeSeriesKMeans(
        n_clusters=2, n_init=n_init, metric="dtw", n_jobs=n_jobs
    ).fit_predict(X_scaled)
    end = time.time()
    time_elapsed = end - start
    average_time_elapsed = float(time_elapsed) / float(n_init)

    print(f"Time for Run {time_elapsed} at average time of {average_time_elapsed}")

    plot_y_dat.append(average_time_elapsed)
    plot_x_dat.append(n_n)

    percent = (float(n_n) / float(n_n_max)) * 100.0
    print(f"{percent:0.01f}% complete")


print("\n" + f"Done! " + "\n" + "Plot!")

y_lin = [x for x in plot_x_dat]
y_quad = [x**2 for x in plot_x_dat]
y_xlogx = [x * np.log(x) for x in plot_x_dat]

coeffs = np.polyfit(plot_x_dat, plot_y_dat, deg=2)
fit = np.poly1d(coeffs)

plt.figure(figsize=(8, 8))
plt.plot(plot_x_dat, plot_y_dat, label="Results", color="black")
plt.plot(plot_x_dat, y_quad, label="Quadratic Fit O(N^2)", color="green")
plt.plot(plot_x_dat, y_quad, label="N Log N Fit O(N Log(N))", color="orange")
plt.plot(plot_x_dat, y_lin, label="Linear Fit O(N)", color="blue")
plt.plot(
    plot_x_dat,
    fit(plot_x_dat),
    label=f"Polynomial Fit {coeffs[0]:0.04f}x^2 + {coeffs[1]:0.04f}x + {coeffs[2]:0.04f}",
    color="red",
    linestyle="-.",
)
plt.ylim(np.min(plot_y_dat), np.max(plot_y_dat))
plt.legend()
plt.xlabel("N Time Series")
plt.ylabel("Run Time [s]")
plt.title(
    "DTW Scaling test - tslearn Package"
    + "\n"
    + f" - {n_t} Temporal Data Points ; {n_q} Parameter Dimensions"
)

# plt.show()
plt.savefig(f"DTW_time_scaling_test.png")
plt.close()
