import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import ascdata

# Use this to do a linear regression which produces a w for each program and breakpoint.
# Features: X, b
# Target: w

### IMPORT DATA ###
print "IMPORTING DATA"

X_raw, y_raw = ascdata.get_asc_data()
X, y = ascdata.remove_zeros(X_raw, y_raw)
# Data isolated for all breakpoints from a specific program.
X_prog1, y_prog1 = ascdata.get_bp_data(1, 0, X, y)
X_prog2, y_prog2 = ascdata.get_bp_data(2, 0, X, y)
X_prog3, y_prog3 = ascdata.get_bp_data(3, 0, X, y)

### GET PREDICTIONS ###
print "GETTING PREDICTIONS"

# Splits into train and test sets, and gets wridge based on train set.
def get_wridge(inX, iny):
    X_train, y_train, X_test, y_test = ascdata.split_train_test(inX, iny)
    #X_train, X_test = normalize_features(X_train, X_test)
    X_train = np.concatenate((X_train, np.ones((X_train.shape[ 0 ], 1))), 1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[ 0 ], 1))), 1)

    nfeatures = X_train.shape[1]

    # Augment X, y
    invT2 = 10
    precision = np.identity(nfeatures) * invT2
    cholesky_precision = np.linalg.cholesky(precision)
    X_train_aug = np.concatenate((X_train, cholesky_precision), axis=0)
    y_train_aug = np.transpose(np.concatenate((y_train, np.zeros(nfeatures))))

    # Get QR decomposition of X
    Q, R = np.linalg.qr(X_train_aug)
    Rinv = np.linalg.inv(R)
    RinvQ = np.dot(Rinv,Q.T)
    wridge = np.dot(RinvQ, y_train_aug)
    return wridge, X_train, y_train, X_test, y_test

wridge_all, X_train_all, y_train_all, X_test_all, y_test_all = get_wridge(X, y)

y_test_pred = np.dot(X_test_all, wridge_all)
rms = sqrt(mean_squared_error(y_test_all, y_test_pred))
print "total rms:", rms

### PLOT PREDICTIONS ###
print "PLOTTING PREDICTIONS"
def plot_predictions(prog_num, bp, inX, iny, wridge, all, test):
    plt.cla()
    X_bp, y_bp = ascdata.get_bp_data(prog_num, bp, inX, iny)
    y_bp_pred = np.dot(X_bp, wridge)
    plot_inputs = X_bp.T[2]
    plt.plot(plot_inputs, y_bp, 'bx')
    plt.plot(plot_inputs, y_bp_pred, 'g')
    plt.savefig("bayes1" + "_" + str(prog_num) + "_" + str(bp) + "_" + test + "_" + all + ".png")
    rms_bp = ascdata.RMSE(y_bp, y_bp_pred)
    print "rms for prognum", prog_num, "at breakpoint", bp, rms_bp

print "Predictions trained on all data"
#plot_predictions(1, 4194659, X_train_all, y_train_all, wridge_all, "all", "total")
plot_predictions(2, 4198375, X_train_all, y_train_all, wridge_all, "all", "total")
#plot_predictions(3, 4194873, X_train_all, y_train_all, wridge_all, "all", "total")

print "Test predictions"
plot_predictions(1, 4194659, X_test_all, y_test_all, wridge_all, "all", "test")
#plot_predictions(2, 4198375, X_test_all, y_test_all, wridge_all, "all", "test")
plot_predictions(3, 4194873, X_test_all, y_test_all, wridge_all, "all", "test")

print "Predictions trained on only program data"
wridge_prog1, X_train_prog1, y_train_prog1, X_test_prog1, y_test_prog1 = get_wridge(X_prog1, y_prog1)
wridge_prog2, X_train_prog2, y_train_prog2, X_test_prog2, y_test_prog2 = get_wridge(X_prog2, y_prog2)
wridge_prog3, X_train_prog3, y_train_prog3, X_test_prog3, y_test_prog3 = get_wridge(X_prog3, y_prog3)

#plot_predictions(1, 4194659, X_train_prog1, y_train_prog1, wridge_prog1, "self", "total")
plot_predictions(2, 4198375, X_train_prog2, y_train_prog2, wridge_prog2, "self", "total")
#plot_predictions(3, 4194873, X_train_prog3, y_train_prog3, wridge_prog3, "self", "total")

print "Test predictions"
plot_predictions(1, 4194659, X_test_prog1, y_test_prog1, wridge_prog1, "self", "test")
#plot_predictions(2, 4198375, X_test_prog2, y_test_prog2, wridge_prog2, "self", "test")
plot_predictions(3, 4194873, X_test_prog3, y_test_prog3, wridge_prog3, "self", "test")