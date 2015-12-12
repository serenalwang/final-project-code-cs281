import ascdata
import numpy as np
from sklearn import gaussian_process
import matplotlib.pyplot as plt

### IMPORT DATA ###
print "IMPORTING DATA"
X_raw, y_raw = ascdata.get_asc_data()
X, y = ascdata.remove_zeros(X_raw, y_raw)
X_train, y_train, X_test, y_test = ascdata.split_train_test(X, y)
X_bp1, y_bp1 = ascdata.get_bp_data(1, 4194659, X, y)
X_bp1_test, y_bp1_test = ascdata.get_bp_data(1, 4194659, X_test, y_test)
X_bp2, y_bp2 = ascdata.get_bp_data(3, 4194873, X, y)
X_bp2_test, y_bp2_test = ascdata.get_bp_data(3, 4194873, X_test, y_test)
X_bp3, y_bp3 = ascdata.get_bp_data(2, 4198375, X, y)
X_bp3_test, y_bp3_test = ascdata.get_bp_data(2, 4198375, X_test, y_test)

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

### PLOT PREDICTIONS ###
print "PLOTTING PREDICTIONS"
def plot_predictions(inX, iny, iny_pred, prog_num, bp, test):
    plt.cla()
    plot_inputs = inX.T[2]
    plt.plot(plot_inputs, iny, c='b', marker='x')
    plt.plot(plot_inputs, iny_pred)
    plt.savefig("gaussianprocess_" + prog_num + "_" + bp + "_" + test + ".png")
    bp_rms = ascdata.RMSE(iny, iny_pred)
    print "rms total for prognum", prog_num, "at breakpoint", bp, bp_rms

y_pred_bp1 = get_predictions(X_bp1)
y_pred_bp1_test = get_predictions(X_bp1_test)
y_pred_bp2 = get_predictions(X_bp2)
y_pred_bp2_test = get_predictions(X_bp2_test)
y_pred_bp3 = get_predictions(X_bp3)
#y_pred_bp3_test = get_predictions(X_bp3_test)

print "RMS for all data for each bp"
plot_predictions(X_bp1, y_bp1, y_pred_bp1, "1", "4194659", "total")
plot_predictions(X_bp2, y_bp2, y_pred_bp2, "3", "4194873", "total")
plot_predictions(X_bp3, y_bp3, y_pred_bp3, "2", "4198375", "total")

print "RMS for test data for each bp"
plot_predictions(X_bp1_test, y_bp1_test, y_pred_bp1_test, "1", "4194659", "test")
plot_predictions(X_bp2_test, y_bp2_test, y_pred_bp2_test, "3", "4194873", "test")
#plot_predictions(X_bp3_test, y_bp3_test, y_pred_bp3_test, "2", "4198375", "test")