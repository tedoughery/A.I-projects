import numpy as np
from matplotlib import pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.
from csv import DictReader
from math import sqrt
from random import randint
from random import uniform


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT:
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename) as file:
        reader = DictReader(file); #open the file
        dictlist = list(reader);

        for row in dictlist:
            val = list(row.values());
            val.pop(0) #remove first element (ID)
            valI = [float(i) for i in val] #convert to float
            dataset.append(valI)
    return np.array(dataset)


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on.
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    #prints the column
    n = dataset.shape[0]
    preMean = 0;
    preDev = 0;
    for row in dataset:
        preMean += row[col]
    mean = preMean/n;
    for row in dataset:
        preDev += pow((row[col] - mean), 2)
    dev = sqrt(preDev/(n-1))
    print(n)
    print('{:.2f}'.format(mean))
    print('{:.2f}'.format(dev))
    pass


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    n = dataset.shape[0]
    mse = 0
    for row in dataset:
        val = betas[0];
        for i in range(len(cols)):
            val += row[cols[i]] * betas[i+1]
        val -= row[0]
        mse += pow(val, 2)
    mse /= n
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    n = dataset.shape[0]
    #for each beta
        #for each row
            #find the value
    temp = 0;
    for i in range(len(betas)):
        foo = 0;
        for row in dataset:
            val = betas[0];
            for j in range(len(cols)):
                val += row[cols[j]] * betas[j+1]
            val -= row[0]
            if i != 0:
                foo += val * row[cols[i-1]]
            else:
                foo += val
        temp = foo * (2/n)
        grads.append(temp)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    #for T times
        #find gradient_descent
        #multiply gradient_descent by eta
        #subtract from the previous beta
    nBetas = betas.copy()

    for i in range(1, T+1):
        dmse = gradient_descent(dataset, cols, nBetas);

        for j in range(len(betas)):
            temp = nBetas[j]
            nBetas[j] = temp - (eta * dmse[j]);

        print(str(i) + " ", end='')
        print(str('{:.2f}'.format(regression(dataset, cols, nBetas)) + " "), end='')
        for j in range(len(nBetas)):
            print(str('{:.2f}'.format(nBetas[j])) + " ", end='')
        print()

def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    betas = []
    mse = None
    y = np.transpose(dataset)[0]
    xT = [[1] * len(dataset)]

    temp = np.transpose(dataset)[cols[0] : cols[len(cols)-1]+1]
    for list in temp:
        xT.append(list.tolist())

    x = np.transpose(xT)
    betas = np.dot(np.dot(np.linalg.inv(np.dot(xT, x)), xT), y)
    mse = regression(dataset, cols, betas)

    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    result = betas[1]
    for i in range(len(features)):
        result += features[i] * betas[i+2]
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    #for n times
        #find y = b0 + b1 * xi +zi
    linear = [None] * len(X)
    z = np.random.normal(0, sigma, len(X))
    for i in range(len(X)):
        y = betas[0] + betas[1] * X[i][0] + z[i]
        linear[i] = [y, X[i][0]]

    quadratic = [None] * len(X)
    z = np.random.normal(0, sigma, len(X)) #more randomness
    for i in range(len(X)):
        y = alphas[0] + alphas[1] * pow(X[i][0], 2) + z[i]
        quadratic[i] = [y, X[i][0]]

    return np.array(linear), np.array(quadratic)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    X = [] #1000 random x such that -100 <= x <= 100
    for i in range(1000):
        X.append([randint(-100, 100)])

    betas = [getNonZeroRand(), getNonZeroRand()] #random nonzero betas and alphas
    alphas = [getNonZeroRand(), getNonZeroRand()]
    sigmas = []
    for i in range(-4, 6):
        sigmas.append(10**i)

    mseLinear = []
    mseQuadratic = []
    for sigma in sigmas:
        linear, quadratic = synthetic_datasets(betas, alphas, X, sigma)

        mse = compute_betas(linear, cols=[1])[0]
        mseLinear.append(mse)

        mse = compute_betas(quadratic, cols=[1])[0]
        mseQuadratic.append(mse)

    lin = plt.plot(sigmas, mseLinear, '-o', label='Linear Dataset')
    quad = plt.plot(sigmas, mseQuadratic, '-o', label='Quadratic Dataset')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Sigma")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('mse.pdf')

def getNonZeroRand(): #helps reduce space taken by plot_mse
    x = 0.0
    while (x == 0.0):
        x = uniform(-25.0, 25.0);
    return x;

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
