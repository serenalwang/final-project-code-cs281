import ascdata
import numpy as np
from sklearn import neural_network
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from math import sqrt

### IMPORT DATA ###
print "IMPORTING DATA"
X_raw, y_raw = ascdata.load_asc_data()
X_nonzero,y_nonzero = ascdata.remove_zeros(X_raw, y_raw)
X,y = ascdata.remove_crazy(X_nonzero,y_nonzero, 100)
X_train, y_train, X_test, y_test = ascdata.split_train_test(X, y)

### GET PREDICTIONS ###
print "GETTING PREDICTIONS"
clf = neural_network.MLPRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

rms_test = ascdata.multi_RMSE(y_test, y_pred)
print "overall rms test", rms_test

score = clf.score(X_test, y_test)
print "overall score", score

# Get cross validation RMS and r^2
def get_kfold_scores(inX, iny, k):
    N = inX.shape[0]
    kf = KFold(N, k, shuffle=True)
    mses=[]
    r2s = []
    for train_index, test_index in kf:
        kf_X_train, kf_X_test = inX[train_index], inX[test_index]
        kf_y_train, kf_y_test = iny[train_index], iny[test_index]
        kf_y_test_pred = clf.predict(kf_X_test)
        mse = ascdata.multi_MSE(kf_y_test, kf_y_test_pred)
        mses.append(mse)
        print "mse:", mse
        r2 = ascdata.multi_r2(kf_y_test, kf_y_test_pred)
        r2s.append(r2)
        print "r2:", r2
    overall_mse = np.mean(mses)
    overall_rmse = sqrt(overall_mse)
    overall_r2 = np.mean(r2s)
    print "KFold cross validation MSE:", overall_mse
    print "KFold cross validation RMSE:", overall_rmse
    print "KFold cross validation r2:", overall_r2

get_kfold_scores(X,y,10)

# del y_pred

### PLOT PREDICTIONS ###
print "PLOTTING PREDICTIONS"
def plot_predictions(prog_num, bp, inX, iny):
    plt.cla()
    X_bp, y_bp = ascdata.get_bp_data(prog_num, bp, inX, iny)
    iny_pred = clf.predict(X_bp)
    plot_inputs = range(1,11)
    plt.plot(plot_inputs, y_bp.flatten(), 'bx')
    plt.plot(plot_inputs, iny_pred.flatten(),'ro')
    plt.savefig("multitask_" + str(prog_num) + "_" + str(bp) + ".png")
    bp_rms = ascdata.multi_RMSE(y_bp, iny_pred)
    bp_r2 = ascdata.multi_r2(y_bp, iny_pred)
    print "rms total for prognum", prog_num, "at breakpoint", bp, bp_rms
    print "r2 total for prognum", prog_num, "at breakpoint", bp, bp_r2

plot_predictions(1, int("400229",16), X, y)

# y_pred_bp1 = get_predictions(X_bp1)
# y_pred_bp1_test = get_predictions(X_bp1_test)
# y_pred_bp2 = get_predictions(X_bp2)
# y_pred_bp2_test = get_predictions(X_bp2_test)
# y_pred_bp3 = get_predictions(X_bp3)
# #y_pred_bp3_test = get_predictions(X_bp3_test)
#
# print "RMS for all data for each bp"
# plot_predictions(X_bp1, y_bp1, y_pred_bp1, "1", "4194659", "total")
# plot_predictions(X_bp2, y_bp2, y_pred_bp2, "3", "4194873", "total")
# plot_predictions(X_bp3, y_bp3, y_pred_bp3, "2", "4198375", "total")
#
# print "RMS for test data for each bp"
# plot_predictions(X_bp1_test, y_bp1_test, y_pred_bp1_test, "1", "4194659", "test")
# plot_predictions(X_bp2_test, y_bp2_test, y_pred_bp2_test, "3", "4194873", "test")
# #plot_predictions(X_bp3_test, y_bp3_test, y_pred_bp3_test, "2", "4198375", "test")