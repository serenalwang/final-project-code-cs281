import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from math import cos, pi
from os import listdir
from os.path import isfile, join
from sklearn.metrics import r2_score

### Methods for reading in ASC data
X_outfile = "X-inputs.npy"
y_outfile = "y-targets.npy"

# Expands the features provided in the asc input file.
# Resulting feature set:
# [program name, breakpoint, overall round number, input parameter number, run number, total rounds,
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

# Store objdump and callgrind data in a dictionary
# key: (program number, ip); value: [objdump data + callgrind ir data]
def get_ip_data():
    objdump_ip_data = np.loadtxt('objdump_ip_features.csv', delimiter = ',', converters={1:lambda s: int(s, 16)})
    callgrind_ir_data = np.loadtxt('callgrind_ir_features.csv', delimiter = ',', converters={1:lambda s: int(s, 16)})
    ip_dict = {}
    n_callgrind_features = 2
    n_objdump_features = 12
    for row in objdump_ip_data:
        key = (row[0], row[1])
        value = row[2:]
        ip_dict[key] = np.concatenate((value, np.zeros(n_callgrind_features)))
    for row in callgrind_ir_data:
        key = (row[0], row[1])
        value = row[2:]
        if key in ip_dict:
            for i in range(len(value)):
                ip_dict[key][i + n_objdump_features] = value[i]
        else:
            ip_dict[key] =  np.concatenate((np.zeros(n_objdump_features),value))
    return ip_dict

# Reads in asc data from files
# Final feature vector: (305)
# [asc features, hexdump features, text features, gprof features, callgrind features, IP features]
#
# asc features: (9)
# [program name, breakpoint, overall round number, input parameter number, run number, total rounds,
#  current round, hamming, mips]

# hexdump features: (256)
# [256 ngrams]

# text features: (3)
# [number of lines, number of words, number of chars]

# program gprof features: (10)
# [Num functions, num function calls, total program time,
#  highest % function time, highest % function calls,
#  variance of function time,  variance of num function calls,
#  max num parents for a function, max num children for a function,
#  num recursive calls]

# program callgrind features: (13)
# [Ir, Dr, Dw, I1mr, d1mr, D1mw, ILmr, DLmr, DLmw, I1missrate, D1missread, D1misswrite, LLmissrate]

# objdump ip features: (12)
# [ is jmp, call, mov, lea, cmp, inc, mul, add, or, push,
#  is target of jmp, distance from target of jmp]

# callgrind ir features: (2)
# [ ir count, ir % ]
def get_asc_data_from_files():
    n_asc_features = 10
    asc_features = get_asc_features()
    y = asc_features[ : , n_asc_features - 1 ]
    X = asc_features[ : , :n_asc_features - 1 ]

    gprof_data = np.loadtxt('program_gprof_features.csv', delimiter = ',')
    callgrind_data = np.loadtxt('program_callgrind_features.csv', delimiter = ',')
    text_data = np.loadtxt('program_text_features.csv', delimiter = ',')
    hexdump_data = np.loadtxt('hexdump_1gram_features.csv', delimiter = ',')

    ip_dict = get_ip_data()
    n_ip_features = 14

    X = X.tolist()
    # Add program features to data matrix
    for i in range(len(X)):
        prog_num = X[i][0]
        ip = X[i][1]
        # Add program features
        X[i] += hexdump_data[prog_num - 1][1:].tolist() + text_data[prog_num - 1][1:].tolist() + gprof_data[prog_num - 1][1:].tolist() + callgrind_data[prog_num - 1][1:].tolist()
        # Add ip features
        ip_features = np.zeros(n_ip_features)
        if (prog_num,ip) in ip_dict:
            ip_features = ip_dict[(prog_num,ip)]
        X[i] += ip_features.tolist()
    X = np.array(X)
    return X, y

# Saves asc data into numpy array files.
def save_asc_data():
    X, y = get_asc_data_from_files()
    np.save(X_outfile,X)
    np.save(y_outfile,y)

# Loads previously saved asc data.
# Returns two numpy arrays.
def load_asc_data():
    X = np.load(X_outfile)
    y = np.load(y_outfile)
    return X, y

# Takes first 10 rounds of training for each IP.
def get_multitask_data(inX, iny):
    outX = []
    outy = []
    last_ip = 0
    nrounds = 0
    cur_y = []
    for i in range(inX.shape[0]):
        cur_ip = inX[i][1]
        if cur_ip == last_ip:
            # Haven't yet filled up all of cur_y yet
            if nrounds < 10:
                if nrounds == 9:
                    cur_y.append(iny[i])
                    assert(len(cur_y) == 10)
                    outX.append(inX[i])
                    outy.append(cur_y)
                    nrounds += 1
                else:
                    cur_y.append(iny[i])
                    nrounds += 1
        # First round of new IP value
        else:
            last_ip = cur_ip
            nrounds = 1
            cur_y = [iny[i]]
    return np.array(outX), np.array(outy)

# Takes first n rounds of training for each IP.
def shrink_data(inX, iny, n):
    outX = []
    outy = []
    last_ip = 0
    nrounds = 0
    for i in range(inX.shape[0]):
        cur_ip = inX[i][1]
        if cur_ip == last_ip:
            # Haven't yet filled up all of cur_y yet
            if nrounds < n:
                outX.append(inX[i])
                outy.append(iny[i])
                nrounds += 1
        # First round of new IP value
        else:
            last_ip = cur_ip
            nrounds = 1
            outX.append(inX[i])
            outy.append(iny[i])
    return np.array(outX), np.array(outy)

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

# Removes all lines of asc data that are above a certain threshold.
# A reasonable threshold is crazy = 100
def remove_crazy(inX, iny, crazy):
    outX = []
    outy = []
    for i in range(inX.shape[0]):
        if iny[i] < crazy:
            outX.append(inX[i])
            outy.append(iny[i])
    return np.array(outX), np.array(outy)

def RMSE(y, y_pred):
    return sqrt(mean_squared_error(y, y_pred))

def multi_RMSE(y, y_pred):
    return sqrt(mean_squared_error(y.flatten(), y_pred.flatten()))

def MSE(y,y_pred):
    return mean_squared_error(y, y_pred)

def multi_MSE(y,y_pred):
    return mean_squared_error(y.flatten(), y_pred.flatten())

def r2(y, y_pred):
    return r2_score(y,y_pred)

def multi_r2(y, y_pred):
    return r2_score(y.flatten(),y_pred.flatten())

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