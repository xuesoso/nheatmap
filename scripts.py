import numpy as np
import pandas as pd

def simulate_data(nrows=120, ncols=60):
    n1 = 0.9
    n2 = 0.1
    lambda_1 = 0.8
    mean2 = 100
    noise1 = 10
    noise2 = 10
    nrow_sub1 = int(nrows/3)
    ncol_sub1 = int(ncols*n1)
    ncol_len1 = int(ncol_sub1 / 3)
    dim1 = (nrow_sub1, ncol_len1)
    nrow_sub2 = int(nrows/3)
    ncol_sub2 = int(ncols*n2)
    ncol_len2 = int(ncol_sub2 / 3)
    dim2 = (nrow_sub2, ncol_len2)
    f1 = np.random.poisson(lambda_1, dim1)*mean2 + np.random.random(dim1)*noise1
    f2 = np.random.poisson(lambda_1, dim1)*mean2 + np.random.random(dim1)*noise1
    f3 = np.random.poisson(lambda_1, dim1)*mean2 + np.random.random(dim1)*noise1
    f4 = np.random.poisson(lambda_1, dim2)*mean2 + np.random.random(dim2)*noise1
    f5 = np.random.poisson(lambda_1, dim2)*mean2 + np.random.random(dim2)*noise1
    f6 = np.random.poisson(lambda_1, dim2)*mean2 + np.random.random(dim2)*noise1
    idx1 = np.random.choice(np.arange(nrows), dim2[0], replace=False)
    np.random.shuffle(idx1)
    idx2 = np.setdiff1d(np.arange(nrows), idx1)
    idx3 = np.random.choice(idx2, dim2[0], replace=False)
    np.random.shuffle(idx3)
    idx2 = np.setdiff1d(idx2, idx3)
    np.random.shuffle(idx2)
    D = np.zeros((nrows, ncols)) + np.random.random((nrows, ncols))*noise2
    D[:dim1[0], :ncol_len1] = f1
    D[dim1[0]:2*dim1[0], ncol_len1:2*ncol_len1] = f2
    D[2*dim1[0]:3*dim1[0], 2*ncol_len1:3*ncol_len1] = f3
    D[idx1, 3*ncol_len1:3*ncol_len1+ncol_len2] = f4
    D[idx2, 3*ncol_len1+ncol_len2:3*ncol_len1+ncol_len2*2] = f5
    D[idx3, 3*ncol_len1+ncol_len2*2:3*ncol_len1+ncol_len2*3] = f6
    Dl = np.log2(D+1)
    D = pd.DataFrame(Dl, index=['sample '+str(x) for x in np.arange(1,
        nrows+1)], columns=['gene '+str(x) for x in np.arange(1, ncols+1)])
    return D
