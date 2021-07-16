import numpy as np
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform, euclidean, cityblock
from fastdtw import fastdtw
from itertools import combinations_with_replacement, product
import time

pi = 3.141596

max_n_n = 300
min_n_n = 10
increment_n_n = 10

n_t = 20


def sin_func(x, l, a):
    y = [a * np.sin(2 * pi * val / l) for val in x]
    return y


def lin_func(x, m, c):
    y = [m * val + c for val in x]
    return y


def DTW_prep(M):
    elements = range(np.shape(M)[0])

    iterator = combinations_with_replacement(elements, r=2)
    D = np.zeros((np.shape(M)[0], np.shape(M)[0]))

    return iterator, D


def DTW_run(M, iterator, D):
    """
    Produce a condensed distance matrix for matrix M using euclidean metric for fastDTW
    https://pypi.org/project/fastdtw/
    \citep{Stan Salvador, and Philip Chan. “FastDTW: Toward accurate dynamic time warping in linear time and space.” Intelligent Data Analysis 11.5 (2007): 561-580.}
    Uses packages:
        import numpy as np
        from scipy.spatial.distance import squareform, euclidean
        from fastdtw import fastdtw
        from itertools import combinations_with_replacement
    Input:
        numpy Matrix shape (n , t) where n is the nth time series of a variable Q, over time series t
    Output:
        numpy Matrix shape (t**2, 2)
    """
    for pair in iterator:

        ii = pair[0]
        jj = pair[1]

        x = M[ii]
        y = M[jj]

        distance, path = fastdtw(x, y, dist=cityblock)

        D[ii, jj] = distance
        D[jj, ii] = distance

    D_compressed = squareform(D)

    return D_compressed


x_data_n_n = []
y_data_t = []

for n_n in np.arange(
    start=min_n_n, stop=max_n_n + increment_n_n, step=increment_n_n, dtype=np.int32
):

    delta_t = 1
    n_t = int(n_t * delta_t)

    # n_n = int((n_n/2)+1)

    print(f"n_t = {n_t}")
    print(f"n_n = {n_n}")

    x = [i for i in np.arange(0, n_t, delta_t)]

    # sin_dat = np.array([sin_func(x,float(n_t)/float(l),100.) for l in np.arange(1,n_n+1,1)])
    lin_dat = np.array([lin_func(x, m, 0.0) for m in np.arange(1, n_n + 1, 1)])

    # print(f"np.shape(sin_dat) = {np.shape(sin_dat)}")
    print(f"np.shape(lin_dat) = {np.shape(lin_dat)}")
    M = lin_dat  # np.concatenate((sin_dat,lin_dat),axis=0)
    print(f"np.shape(M) = {np.shape(M)}")

    iterator, D = DTW_prep(M)

    start = time.time()
    D = DTW_run(M, iterator, D)
    end = time.time()
    elapsed = end - start
    print(f"Elapsed time in DTW = {elapsed}s")

    x_data_n_n.append(n_n)
    y_data_t.append(elapsed)

y_lin = [x for x in x_data_n_n]
y_quad = [x ** 2 for x in x_data_n_n]
y_xlogx = [x * np.log(x) for x in x_data_n_n]

coeffs = np.polyfit(x_data_n_n, y_data_t, deg=2)
fit = np.poly1d(coeffs)


plt.figure()
plt.plot(x_data_n_n, y_data_t, label="Results", color="black")
plt.plot(x_data_n_n, y_quad, label="Quadratic Fit O(N^2)", color="green")
plt.plot(x_data_n_n, y_quad, label="N Log N Fit O(N Log(N))", color="orange")
plt.plot(x_data_n_n, y_lin, label="Linear Fit O(N)", color="blue")
plt.plot(
    x_data_n_n,
    fit(x_data_n_n),
    label=f"Polynomial Fit {coeffs[0]:0.04f}x^2 + {coeffs[1]:0.04f}x + {coeffs[2]:0.04f}",
    color="red",
    linestyle="-.",
)
plt.ylim(np.min(y_data_t), np.max(y_data_t))
plt.legend()
plt.xlabel("N Time Series")
plt.ylabel("Single CPU Run Time [s]")
plt.title(
    "DTW Scaling test - FastDTW Package" + "\n" + "Single CPU, Reduced Matrix Method"
)

# plt.show()
plt.savefig(f"DTW_time_scaling_test_v2_n_t-{int(n_t)}.pdf")
plt.close()

# In[15]:


# Z = linkage(D,method='single')
# fig = plt.figure(figsize=(15,20))
# dn = dendrogram(Z)


# In[16]:


# max_d = 500
# clusters = fcluster(Z, max_d, criterion='distance')
# clusters


# In[17]:


# class_num = 1
# select = M[np.where(clusters==class_num)]
# for jj in range(np.shape(select)[0]):
#     plt.plot(x,select[jj,:])


# In[18]:


# class_num = 2
# select = M[np.where(clusters==class_num)]
# for jj in range(np.shape(select)[0]):
#     plt.plot(x,select[jj,:])
