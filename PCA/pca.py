from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # TODO: add your code here
    x = np.load(filename);
    x = x-np.mean(x, axis=0)
    return x

def get_covariance(dataset):
    # TODO: add your code here
    x = dataset
    xt = np.transpose(x)
    num_rows, num_cols = x.shape
    retVal = (1/(num_rows-1)) * np.dot(xt, x)
    return retVal

def get_eig(S, m):
    # TODO: add your code here
    N = np.shape(S)[0]
    x, y = eigh(S, subset_by_index=[N-m, N-1])
    x = np.flip(x)
    return np.diag(x), np.flip(y, axis=1)

def get_eig_perc(S, perc):
    # TODO: add your code here
    eigenSum = eigh(S)[0].sum()
    x, y = eigh(S/eigenSum, subset_by_value=[perc, np.inf])
    x *= eigenSum
    x = np.flip(x)
    return np.diag(x), np.flip(y, axis=1)

def project_image(img, U):
    # TODO: add your code here

    #treat a as a matrix
    m  = np.shape(U)[0]
    ut = np.transpose(U);
    a = np.dot(ut, img);
    retval = np.dot(U, a)
    return retval

def display_image(orig, proj):
    # TODO: add your code here
    #reshape to be 32x32
    #create a figure with 2 subplots
    a = np.reshape(orig, (32, 32), order='F')
    b = np.reshape(proj, (32, 32), order='F')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5,5.5))
    fig.tight_layout()

    o = ax1.imshow(a, aspect='equal')
    ax1.title.set_text('Original')
    fig.colorbar(o, ax=ax1, shrink=1.0, fraction=0.0452)

    p = ax2.imshow(b, aspect='equal')
    ax2.title.set_text('Projection')
    fig.colorbar(p, ax=ax2, shrink=1.0, fraction=0.0452)

    plt.show()
    return
