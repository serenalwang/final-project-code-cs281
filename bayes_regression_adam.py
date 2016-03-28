from __future__ import absolute_import
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad

from optimizers import adam
import ascdata

# Use this to do a linear regression which produces a w for each program and breakpoint.
# Features: X, b
# Target: w

### IMPORT DATA ###
print "IMPORTING DATA"

# X_raw, y_raw = ascdata.load_asc_data()
# X, y = ascdata.remove_zeros(X_raw, y_raw)

X,y = ascdata.load_shrunken_asc_data()

# Data isolated for all breakpoints from a specific program.
# X_prog1, y_prog1 = ascdata.get_bp_data(1, 0, X, y)
# X_prog2, y_prog2 = ascdata.get_bp_data(2, 0, X, y)
# X_prog3, y_prog3 = ascdata.get_bp_data(3, 0, X, y)

D = 1
d = 10
A_phi = ascdata.generate_A(d, D)
b_phi = ascdata.generate_b(d)

### GET PREDICTIONS ###
print "GETTING PREDICTIONS"

# Returns a N by K matrix W where W[i] corresponds with s[i]
# Takes in raw s and not phi(s)
def get_predictions(Aopt, inX, s_bp):
    s_bp_phi = ascdata.generate_phi(s_bp, d, A_phi, b_phi)
    y_pred = []
    for i in range(inX.shape[0]):
        wi = np.dot(Aopt, inX[i])
        y_pred.append(np.dot(wi.T, s_bp_phi[i]))
    return y_pred

# inX should not contain s column.
# ins should be raw s and not phi(s)
def plot_predictions(prog_num, bp, inX, iny, ins, Aopt, test):
    plt.cla()
    X_bp, y_bp, s_bp = ascdata.get_bp_data_s(prog_num, bp, inX, iny, ins)
    y_bp_pred = get_predictions(Aopt, X_bp, s_bp)
    plot_inputs = s_bp
    plt.plot(plot_inputs, y_bp, 'bx')
    plt.plot(plot_inputs, y_bp_pred, 'gx')
    plt.draw()
    plt.savefig("bayes2exp" + "_" + str(prog_num) + "_" + str(bp) + "_" + test + ".png")
    rms_bp = ascdata.RMSE(y_bp, y_bp_pred)
    print "rms for prognum", prog_num, "at breakpoint", bp, rms_bp

# Splits into train and test sets, and gets Aopt based on train set.
# Expects inX
def get_Aopt(inX, iny):
    X_train, y_train, X_test, y_test = ascdata.split_train_test(inX, iny)
    X_train = np.concatenate((X_train, np.ones((X_train.shape[ 0 ], 1))), 1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[ 0 ], 1))), 1)
    X_train_less, s_train = ascdata.split_X_s(X_train)
    X_test_less, s_test = ascdata.split_X_s(X_test)

    s_train_phi = ascdata.generate_phi(s_train, d, A_phi, b_phi)
    s_test_phi = ascdata.generate_phi(s_test, d, A_phi, b_phi)

    nfeatures = X_train.shape[1] - 1
    # Dimensions of phi(s)
    nfeatures_phi = d
    invT2 = 10

    def logprob(inA, inX, iny, ins_phi):
        RMS = 0
        for i in range(len(iny)):
            wi = np.dot(inA, inX[i])
            RMS_current = (iny[i] - np.dot(wi, ins_phi[i]))**2
            RMS += RMS_current
        return -RMS

    objective = lambda inA, t: -logprob(inA, X_train_less, y_train, s_train_phi)

    LLHs = []
    LLH_xs = []

    def callback(params, t, g):
        LLH = -objective(params, t)
        LLHs.append(LLH)
        LLH_xs.append(t)
        print("Iteration {} log likelihood {}".format(t, LLH))

    init_A = 0.00000000001*(np.ones((nfeatures_phi, nfeatures)))
    # init_A =  [[ -3.05236728e-04,  -9.50015728e-04,  -3.80139503e-04,   1.44010470e-04, -3.05236728e-04,
    #              -4.96117987e-04,  -1.02736409e-04,  -1.86416292e-04, -9.52628589e-04,  -1.55023279e-03,
    #              1.44717581e-04,   1.00000000e-11, -9.50028200e-04,  -4.96117987e-04,   1.00000000e-11,
    #              -3.05236728e-04, 1.77416412e-06,  -8.16665436e-06,   3.12622951e-05,  -8.25700143e-04,
    #              1.44627987e-04,   1.90211243e-05,  -8.28273186e-04,  -9.41349990e-04, -4.56671031e-04,
    #              9.79097070e-03,  -6.41866046e-04,  -7.79274856e-05, 1.44539330e-04,  -3.05236728e-04,
    #              -5.99188450e-04,  -7.29470175e-04, -6.69558174e-04,  -9.50028200e-04]]
    init_A = np.array(init_A)

    print("Optimizing network parameters...")
    optimized_params = adam(grad(objective), init_A,
                            step_size=0.01, num_iters=1000, callback=callback)

    Aopt = optimized_params
    print "Aopt = ", Aopt

    return Aopt, X_train_less, y_train, s_train, X_test_less, y_test, s_test, LLHs, LLH_xs

Aopt_all, X_train_all, y_train_all, s_train_all, X_test_all, y_test_all, s_test_all, LLHs, LLH_xs = get_Aopt(X, y)

# ### PLOT PREDICTIONS ###
print "PLOTTING PREDICTIONS"

y_test_pred = y_bp_pred = get_predictions(Aopt_all, X_test_all, s_test_all)
rms = ascdata.RMSE(y_test_all, y_test_pred)
print "total rms:", rms

r2 = ascdata.r2(y_test_all, y_test_pred)
print "total r2:", r2

# # Get cross validation RMS
# def get_kfold_scores(inX, iny, k):
#     N = inX.shape[0]
#     kf = KFold(N, k, shuffle=True)
#     mses=[]
#     r2s=[]
#     for train_index, test_index in kf:
#         kf_X_train, kf_X_test = inX[train_index], inX[test_index]
#         kf_y_train, kf_y_test = iny[train_index], iny[test_index]
#         kf_X_train = np.concatenate((kf_X_train, np.ones((kf_X_train.shape[ 0 ], 1))), 1)
#         kf_X_test = np.concatenate((kf_X_test, np.ones((kf_X_test.shape[ 0 ], 1))), 1)
#         X_test_less, s_test = ascdata.split_X_s(X_test)
#         wridge = get_wridge_given_sets(kf_X_train, kf_y_train, kf_X_test, kf_y_test)
#         kf_y_test_pred = np.dot(kf_X_test, wridge)
#         mse = mean_squared_error(kf_y_test, kf_y_test_pred)
#         mses.append(mse)
#         print "mse:", mse
#         r2 = ascdata.r2(kf_y_test, kf_y_test_pred)
#         r2s.append(r2)
#         print "r2:", r2
#     overall_r2 = np.mean(r2s)
#     overall_mse = np.mean(mses)
#     overall_rmse = sqrt(overall_mse)
#     print "KFold cross validation MSE:", overall_mse
#     print "KFold cross validation RMSE:", overall_rmse
#     print "KFold cross validation r2:", overall_r2
#
# get_kfold_scores(X,y,10)

# Set up plot
fig = plt.figure(1, facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.show(block=False)

print "RMS for all data for each bp"
plot_predictions(1, 4194659, X_train_all, y_train_all, s_train_all, Aopt_all, "total")
# plot_predictions(3, 4194873, X_train_all, y_train_all, s_train_all, Aopt_all, "total")

print "RMS for test data for each bp"
plot_predictions(1, 4194659, X_test_all, y_test_all, s_test_all, Aopt_all, "test")
# plot_predictions(3, 4194873, X_test_all, y_test_all, s_test_all, Aopt_all, "test")

plt.figure(2, facecolor='white')
plt.plot(LLH_xs, LLHs)
plt.savefig('bayes2_LLH.png')