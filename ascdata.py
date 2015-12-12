import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from math import cos, pi

### Methods for reading in ASC data

# Reads in asc data from files
def get_asc_data():
    n_asc_features = 6
    asc_data1 = np.loadtxt('program_asc_features.csv', delimiter = ',', skiprows = 1)
    asc_data2 = np.loadtxt('program_asc_features_temp.csv', delimiter = ',', skiprows = 1)

    asc_data = np.concatenate((asc_data1,asc_data2))
    y = asc_data[ : , n_asc_features - 1 ]
    X = asc_data[ : , :n_asc_features - 1 ]

    gprof_data = np.loadtxt('program_gprof_features.csv', delimiter = ',', skiprows = 1)
    callgrind_data = np.loadtxt('program_callgrind_features.csv', delimiter = ',', skiprows = 1)
    text_data = np.loadtxt('program_text_features.csv', delimiter = ',', skiprows = 1)

    X = X.tolist()
    # Add static features to data matrix
    for i in range(len(X)):
        prog_num = X[i][0]
        X[i] += gprof_data[prog_num - 1].tolist() + callgrind_data[prog_num - 1].tolist() + text_data[prog_num - 1].tolist()
    X = np.array(X)
    return X, y

# Output X and y arrays for the bp and prog_num given.
# Filters theses from inX and iny.
def get_bp_data(prog_num, bp, inX, iny):
    outX = []
    outy = []
    for i in range(inX.shape[0]):
        if bp != 0:
            if inX[i][0] == prog_num and inX[i][1] == bp:
                outX.append(inX[i])
                outy.append(iny[i])
        else:
             if inX[i][0] == prog_num:
                outX.append(inX[i])
                outy.append(iny[i])
    return np.array(outX), np.array(outy)

# Output X, y, s arrays for the bp and prog_num given.
# Filters these from inX, iny, and ins.
def get_bp_data_s(prog_num, bp, inX, iny, ins):
    outX = []
    outy = []
    outs = []
    for i in range(inX.shape[0]):
        if bp != 0:
            if inX[i][0] == prog_num and inX[i][1] == bp:
                outX.append(inX[i])
                outy.append(iny[i])
                outs.append(ins[i])
        else:
             if inX[i][0] == prog_num:
                outX.append(inX[i])
                outy.append(iny[i])
                outs.append(ins[i])
    return np.array(outX), np.array(outy), np.array(outs)

# Removes all lines of asc data that contain zeros.
def remove_zeros(inX, iny):
    outX = []
    outy = []
    for i in range(inX.shape[0]):
        if iny[i] != 0:
            outX.append(inX[i])
            outy.append(iny[i])
    return np.array(outX), np.array(outy)

def RMSE(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))

# Splits train and test set at random.
def split_train_test(X, y, fraction_train = 9.0 / 10.0):
    ndata = X.shape[0]
    trainindices = np.random.choice(ndata, round(ndata * fraction_train), replace=False)
    testindices = []
    for i in range(ndata):
        if i not in trainindices:
            testindices.append(i)
    X_train = X[trainindices]
    y_train = y[trainindices]
    X_test = X[testindices]
    y_test = y[testindices]
    return X_train, y_train, X_test, y_test

def split_X_s(inX):
    s = inX.T[2]
    inX = np.concatenate((inX.T[:2], inX.T[3:])).T
    return inX, s

# Generates the d by D matrix A such that each row of A is sampled from N(0, I)
def generate_A(d, D):
    A = np.zeros((d,D))
    for i in range(d):
        a = np.random.multivariate_normal(np.zeros(D), np.identity(D))
        A[i] = a
    return A

# Generates the d by 1 mvector b such that each entry in b is sampled from U[0, 2pi]
def generate_b(d):
    b = np.zeros(d)
    for i in range(d):
        b[i] = np.random.uniform(0, 2 * pi)
    return b

# Generates phi(s)
def generate_phi(ins, d, A, b):
    print "Generating phi(s) for dimension d = ", d
    N = ins.shape[0]
    phi = np.zeros((N, d))
    for i in range(N):
        phi[i] = np.dot(A.T, ins[i]) + b
        # take cos of all elements if phi
        for j in range(d):
            phi[i][j] = cos(phi[i][j])
    print "Done generating phi for dimension d = ", d
    return phi