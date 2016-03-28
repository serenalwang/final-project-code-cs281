import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from math import sqrt
import matplotlib.pyplot as plt
import ascdata

# Use this to do a linear regression which produces a w for each program and breakpoint.
# Features: X, b
# Target: w

### IMPORT DATA ###
print "IMPORTING DATA"

X, y = ascdata.load_nonzero_asc_data()

X_prog1, y_prog1, X_prog2, y_prog2, X_prog3, y_prog3 = ascdata.load_nonzero_progs()
X_noprog1, y_noprog1, X_noprog2, y_noprog2, X_noprog3, y_noprog3 = ascdata.load_nonzero_noprogs()

# # Data isolated for all breakpoints from a specific program.
# X_prog1, y_prog1 = ascdata.get_bp_data(5, 0, X, y)
# X_prog2, y_prog2 = ascdata.get_bp_data(1, 0, X, y)
# X_prog3, y_prog3 = ascdata.get_bp_data(4, 0, X, y)

# Plot rmses for each lambda
RMSE_reduced_lambdas = [18.12, 18.11, 18.14, 18.35, 18.54, 19.07, 20.08]
RMSE_full_lambdas = [6862.26, 6863.32, 6862.94, 6877.66, 7001.80, 7323.43,7430.91]

r2_reduced_lambdas = [0.37, 0.37, 0.37, 0.36, 0.34, 0.30, 0.23]
r2_full_lambdas = [0.35,0.35, 0.35, 0.35, 0.32, 0.26,0.24]

def plot_lambdas():
    plt.cla()
    w = 0.3
    inputs_l = range(len(r2_reduced_lambdas))
    # inputs_r = [x + w for x in inputs_l]
    # plt.bar(inputs_l, r2_reduced_lambdas,width=w,color='b',align='center')
    # plt.bar(inputs_r, r2_full_lambdas,width=w,color='g',align='center')
    # plt.autoscale(tight=True)
    # plt.savefig("bayes1" + "_" + "lambdas_r2" + ".png")

    # Plot RMSEs
    plt.cla()
    plt.bar(inputs_l, RMSE_reduced_lambdas,width=w,color='b',align='center')
    plt.savefig("bayes1" + "_" + "lambdas_reduced_RMSE" + ".png")
    plt.cla()
    plt.bar(inputs_l, RMSE_full_lambdas,width=w,color='g',align='center')
    plt.savefig("bayes1" + "_" + "lambdas_full_RMSE" + ".png")


# plot_lambdas()

### GET PREDICTIONS ###
print "GETTING PREDICTIONS"
invT2 = 10

# Splits into train and test sets, and gets wridge based on train set.
def get_wridge(inX, iny):
    X_train, y_train, X_test, y_test = ascdata.split_train_test(inX, iny)
    X_train = np.concatenate((X_train, np.ones((X_train.shape[ 0 ], 1))), 1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[ 0 ], 1))), 1)

    nfeatures = X_train.shape[1]

    # Augment X, y
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

# gets wridge without splitting into train and test sets. Instead uses sets passed in.
def get_wridge_given_sets(inX_train, iny_train):
    #X_train, X_test = normalize_features(X_train, X_test)
    nfeatures = inX_train.shape[1]

    # Augment X, y
    precision = np.identity(nfeatures) * invT2
    cholesky_precision = np.linalg.cholesky(precision)
    X_train_aug = np.concatenate((inX_train, cholesky_precision), axis=0)
    y_train_aug = np.transpose(np.concatenate((iny_train, np.zeros(nfeatures))))

    # Get QR decomposition of X
    Q, R = np.linalg.qr(X_train_aug)
    Rinv = np.linalg.inv(R)
    RinvQ = np.dot(Rinv,Q.T)
    wridge = np.dot(RinvQ, y_train_aug)
    return wridge

wridge_all, X_train_all, y_train_all, X_test_all, y_test_all = get_wridge(X, y)

y_test_pred = np.dot(X_test_all, wridge_all)
rms = ascdata.RMSE(y_test_all, y_test_pred)
print "total rms:", rms
r2 = ascdata.r2(y_test_all, y_test_pred)
print "total r2:", r2

def plot_feature_weights(wridge):
    plt.cla()

    # Properly map xticks
    # Sort feature weights by feature number
    tuple_list = [(wridge[i], i) for i in range(len(wridge) - 1)]
    sorted_tuple_list = sorted(tuple_list)

    # Now, we need to sort the featurenames
    featurenames = ascdata.get_feature_names()
    sorted_featurenames = []
    sorted_w = []
    for w_val, i in sorted_tuple_list:
        if w_val > 0.05 or w_val < -0.05:
            sorted_featurenames.append(featurenames[i])
            sorted_w.append(w_val)

    inputs = range(len(sorted_w))
    plt.figure(figsize=(20,10))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.bar(inputs,sorted_w)
    plt.xticks( inputs, sorted_featurenames, rotation=70, fontsize=11 )
    plt.savefig("bayes1" + "_" + "featureweights" + ".png")

# plot_feature_weights(wridge_all)

def plot_actualvpredicted(y, y_pred):
    plt.cla()
    plt.scatter(y_pred, y)
    ones = range(int(max(y_pred)))
    plt.plot(ones, ones)
    plt.savefig("bayes1" + "_" + "actualvpredicted" + "_" + str(invT2) + ".png")

# plot_actualvpredicted(y_test_all, y_test_pred)

# Get cross validation RMS
def get_kfold_scores(inX, iny, k):
    N = inX.shape[0]
    kf = KFold(N, k, shuffle=True)
    mses=[]
    r2s=[]
    for train_index, test_index in kf:
        kf_X_train, kf_X_test = inX[train_index], inX[test_index]
        kf_y_train, kf_y_test = iny[train_index], iny[test_index]
        kf_X_train = np.concatenate((kf_X_train, np.ones((kf_X_train.shape[ 0 ], 1))), 1)
        kf_X_test = np.concatenate((kf_X_test, np.ones((kf_X_test.shape[ 0 ], 1))), 1)
        wridge = get_wridge_given_sets(kf_X_train, kf_y_train, kf_X_test, kf_y_test)
        kf_y_test_pred = np.dot(kf_X_test, wridge)
        mse = mean_squared_error(kf_y_test, kf_y_test_pred)
        mses.append(mse)
        print "mse:", mse
        r2 = ascdata.r2(kf_y_test, kf_y_test_pred)
        r2s.append(r2)
        print "r2:", r2
    overall_r2 = np.mean(r2s)
    overall_mse = np.mean(mses)
    overall_rmse = sqrt(overall_mse)
    print "KFold cross validation MSE:", overall_mse
    print "KFold cross validation RMSE:", overall_rmse
    print "KFold cross validation r2:", overall_r2

# get_kfold_scores(X,y,10)

### PLOT PREDICTIONS ###
print "PLOTTING PREDICTIONS"
def plot_predictions(prog_num, bp, inX, iny, wridge, all, test):
    plt.cla()
    X_bp, y_bp = ascdata.get_bp_data(prog_num, bp, inX, iny)
    y_bp_pred = np.dot(X_bp, wridge)
    plot_inputs = X_bp.T[2]
    plt.plot(plot_inputs, y_bp, 'b^')
    plt.plot(plot_inputs, y_bp_pred, 'go')
    plt.savefig("bayes1" + "_" + str(prog_num) + "_" + str(bp) + "_" + test + "_" + all + ".png")
    rms_bp = ascdata.RMSE(y_bp, y_bp_pred)
    print "rms for prognum", prog_num, "at breakpoint", bp, rms_bp
    r2_bp = ascdata.r2(y_bp,y_bp_pred)
    print "r2 score for prognum", prog_num, "at breakpoint", bp, r2_bp

print "Predictions trained on all data"
#plot_predictions(5, int("40024f",16) , X_train_all, y_train_all, wridge_all, "all", "total")
#plot_predictions(1, int("4014d2",16), X_train_all, y_train_all, wridge_all, "all", "total")
plot_predictions(4, int("400649",16), X_train_all, y_train_all, wridge_all, "all", "total")

# print "Test predictions"
# plot_predictions(1, 4194659, X_test_all, y_test_all, wridge_all, "all", "test")
# plot_predictions(2, 4198375, X_test_all, y_test_all, wridge_all, "all", "test")
# plot_predictions(3, 4194873, X_test_all, y_test_all, wridge_all, "all", "test")
#
print "Predictions trained on only program data"
# wridge_prog1, X_train_prog1, y_train_prog1, X_test_prog1, y_test_prog1 = get_wridge(X_prog1, y_prog1)
# wridge_prog2, X_train_prog2, y_train_prog2, X_test_prog2, y_test_prog2 = get_wridge(X_prog2, y_prog2)
wridge_prog3, X_train_prog3, y_train_prog3, X_test_prog3, y_test_prog3 = get_wridge(X_prog3, y_prog3)
#
# plot_predictions(5, int("40024f",16), X_train_prog1, y_train_prog1, wridge_prog1, "self", "total")
# plot_predictions(1, int("4014d2",16), X_train_prog2, y_train_prog2, wridge_prog2, "self", "total")
plot_predictions(4, int("400649",16), X_train_prog3, y_train_prog3, wridge_prog3, "self", "total")
#
# print "Test predictions"
# plot_predictions(1, 4194659, X_test_prog1, y_test_prog1, wridge_prog1, "self", "test")
# #plot_predictions(2, 4198375, X_test_prog2, y_test_prog2, wridge_prog2, "self", "test")
# plot_predictions(3, 4194873, X_test_prog3, y_test_prog3, wridge_prog3, "self", "test")

print "Predictions trained on all but program data"
# Augment noprogs

# X_noprog1 = np.concatenate((X_noprog1, np.ones((X_noprog1.shape[ 0 ], 1))), 1)
# X_noprog2 = np.concatenate((X_noprog2, np.ones((X_noprog2.shape[ 0 ], 1))), 1)
X_noprog3 = np.concatenate((X_noprog3, np.ones((X_noprog3.shape[ 0 ], 1))), 1)
# wridge_noprog1 = get_wridge_given_sets(X_noprog1, y_noprog1)
# wridge_noprog2 = get_wridge_given_sets(X_noprog2, y_noprog2)
wridge_noprog3 = get_wridge_given_sets(X_noprog3, y_noprog3)
#
# plot_predictions(5, int("40024f",16), X_train_prog1, y_train_prog1, wridge_noprog1, "noprog", "total")
# plot_predictions(1, int("4014d2",16), X_train_prog2, y_train_prog2, wridge_noprog2, "noprog", "total")
plot_predictions(4, int("400649",16), X_train_prog3, y_train_prog3, wridge_noprog3, "noprog", "total")