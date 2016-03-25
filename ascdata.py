import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from math import cos, pi
from os import listdir
from os.path import isfile, join

### Methods for reading in ASC data

# Expands the features provided in the asc input file.
# Resulting feature set:
# [program name, breakpoint, overall round number, input parameter, run number, total rounds,
#         current round, hamming, mips, logloss]
def expand_asc_features(cur_data):
    expanded_data = np.empty
    input_number = 0
    cur_input = 0
    initialized_expanded_data = False
    for i in range(cur_data.shape[0]):
        new_row = cur_data[i]
        new_input = int(new_row[2])
        if new_input != cur_input:
            input_number += 1
            cur_input = new_input
        # Replace random input with input number
        new_row[2] = input_number
        # Add overall round number
        new_row = np.insert(new_row,2,i + 1)
        if not initialized_expanded_data:
            expanded_data = new_row
            initialized_expanded_data = True
        else:
            expanded_data = np.vstack((expanded_data, new_row))
    return expanded_data

# Processes asc_data file and cleans up features
def get_asc_features():
    asc_data = np.empty
    initialized_asc_data = False
    ascfeatures_dir = "ascfeatures/"
    ascfeatures_files = [f for f in listdir(ascfeatures_dir) if isfile(join(ascfeatures_dir, f))]
    # ascfeatures_files = ["1-4002bd-1-asc.csv"]
    for f in ascfeatures_files:
        cur_data = np.loadtxt(join(ascfeatures_dir, f), delimiter = ',', converters={1:lambda s: int(s, 16)})
        expanded_data = expand_asc_features(cur_data)
        if not initialized_asc_data:
            asc_data = expanded_data
            initialized_asc_data = True
        else:
            asc_data = np.concatenate((asc_data,expanded_data))
    return asc_data


# Reads in asc data from files
def get_asc_data():
    n_asc_features = 10
    asc_features = get_asc_features()
    y = asc_features[ : , n_asc_features - 1 ]
    X = asc_features[ : , :n_asc_features - 1 ]

    # gprof_data = np.loadtxt('program_gprof_features.csv', delimiter = ',', skiprows = 1)
    # callgrind_data = np.loadtxt('program_callgrind_features.csv', delimiter = ',', skiprows = 1)
    text_data = np.loadtxt('program_text_features.csv', delimiter = ',', skiprows = 1)
    hexdump_data = np.loadtxt('hexdump_1gram_features.csv', delimiter = ',')

    X = X.tolist()
    # Add static features to data matrix
    for i in range(len(X)):
        prog_num = X[i][0]
        X[i] += hexdump_data[prog_num - 1].tolist() + text_data[prog_num - 1].tolist()
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
    if len(outX) == 0:
        print "WARNING: get_bp_data: no data found for bp", bp
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