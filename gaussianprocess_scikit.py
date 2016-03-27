import ascdata
import numpy as np
from sklearn import gaussian_process
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from math import sqrt
from tempfile import mkdtemp
import os.path as path

### IMPORT DATA ###
print "IMPORTING DATA"
X_raw, y_raw = ascdata.load_asc_data()
print "done loading data"
X,y = ascdata.remove_zeros(X_raw, y_raw)
print "done removing zeros"
del X_raw, y_raw
# X,y = ascdata.remove_crazy(X_nonzero,y_nonzero, 100)
# print "done removing noncrazy"
# X, y = ascdata.shrink_data(X_nonzero, y_nonzero, 2)
X_train, y_train, X_test, y_test = ascdata.split_train_test(X, y)
# print "done splitting data"
# X_bp1, y_bp1 = ascdata.get_bp_data(1, int("400229",16), X_train, y_train)
# X_bp1_test, y_bp1_test = ascdata.get_bp_data(1, int("400229",16), X_test, y_test)
# del X_train, y_train, X_test, y_test

# filename = path.join(mkdtemp(), 'newfile.dat')
# fp_X_train = np.memmap(filename, dtype='float32', mode='w+', shape=X_train.shape)
# fp_X_train[:] = X_train[:]
#
# del X_train

# X_bp1_test, y_bp1_test = ascdata.get_bp_data(1, 4194659, X_test, y_test)
# X_bp2, y_bp2 = ascdata.get_bp_data(3, 4194873, X, y)
# X_bp2_test, y_bp2_test = ascdata.get_bp_data(3, 4194873, X_test, y_test)
# X_bp3, y_bp3 = ascdata.get_bp_data(2, 4198375, X, y)
# X_bp3_test, y_bp3_test = ascdata.get_bp_data(2, 4198375, X_test, y_test)

### GET PREDICTIONS ###
print "GETTING PREDICTIONS"
gp = gaussian_process.GaussianProcess(theta0=1e-1, thetaL=1e-3, thetaU=1, nugget=y_train+20)
gp.fit(X_train, y_train)
# Returns predictions for inXtest after training on inXtrain and inytrain.
def get_predictions(inXtest):
    y_pred, sigma2_pred = gp.predict(inXtest, eval_MSE=True)
    return y_pred

y_pred_test = get_predictions(X_test)

rms_test = ascdata.RMSE(y_test, y_pred_test)
print "rms test", rms_test

del y_pred_test

# Get cross validation RMS and r^2
def get_kfold_scores(inX, iny, k):
    N = inX.shape[0]
    kf = KFold(N, k, shuffle=True)
    mses=[]
    r2s = []
    for train_index, test_index in kf:
        kf_X_train, kf_X_test = inX[train_index], inX[test_index]
        kf_y_train, kf_y_test = iny[train_index], iny[test_index]
        kf_y_test_pred = gp.predict(kf_X_test)
        mse = ascdata.MSE(kf_y_test, kf_y_test_pred)
        mses.append(mse)
        print "mse:", mse
        r2 = ascdata.r2(kf_y_test, kf_y_test_pred)
        r2s.append(r2)
        print "r2:", r2
    overall_mse = np.mean(mses)
    overall_rmse = sqrt(overall_mse)
    overall_r2 = np.mean(r2s)
    print "KFold cross validation MSE:", overall_mse
    print "KFold cross validation RMSE:", overall_rmse
    print "KFold cross validation r2:", overall_r2

get_kfold_scores(X,y,10)

### PLOT PREDICTIONS ###
print "PLOTTING PREDICTIONS"
def plot_predictions(inX, iny, iny_pred, prog_num, bp, test):
    plt.cla()
    plot_inputs = inX.T[2]
    plt.plot(plot_inputs, iny, 'bx')
    plt.plot(plot_inputs, iny_pred, 'ro')
    plt.savefig("gaussianprocess_" + str(prog_num) + "_" + str(bp) + "_" + test +".png")
    bp_rms = ascdata.RMSE(iny, iny_pred)
    print "rms total for prognum", prog_num, "at breakpoint", bp, bp_rms
    bp_r2 = ascdata.r2(iny, iny_pred)
    print "r2 total for prognum", prog_num, "at breakpoint", bp, bp_r2

X_bp1, y_bp1 = ascdata.get_bp_data(1, int("400229",16), X, y)
X_bp1_test, y_bp1_test = ascdata.get_bp_data(1, int("400229",16), X_test, y_test)
y_pred_bp1 = get_predictions(X_bp1)
y_pred_bp1_test = get_predictions(X_bp1_test)
# # y_pred_bp2 = get_predictions(X_bp2)
# # y_pred_bp2_test = get_predictions(X_bp2_test)
# # y_pred_bp3 = get_predictions(X_bp3)
# #y_pred_bp3_test = get_predictions(X_bp3_test)
#
# print "RMS for all data for each bp"
plot_predictions(X_bp1, y_bp1, y_pred_bp1, 1, int("400229",16), "total")
# plot_predictions(X_bp2, y_bp2, y_pred_bp2, "3", "4194873", "total")
# plot_predictions(X_bp3, y_bp3, y_pred_bp3, "2", "4198375", "total")

print "RMS for test data for each bp"
plot_predictions(X_bp1_test, y_bp1_test, y_pred_bp1_test, 1, int("400229",16), "test")
# plot_predictions(X_bp2_test, y_bp2_test, y_pred_bp2_test, "3", "4194873", "test")
#plot_predictions(X_bp3_test, y_bp3_test, y_pred_bp3_test, "2", "4198375", "test")