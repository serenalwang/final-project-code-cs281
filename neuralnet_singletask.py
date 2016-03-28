from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad

from optimizers import adam
import ascdata

def make_nn_funs(layer_sizes, weight_scale=10.0, noise_scale=0.1, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron."""
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        for m, n in shapes:
            cur_layer_weights = weights[:m*n]     .reshape((m, n))
            cur_layer_biases  = weights[m*n:m*n+n].reshape((1, n))
            yield cur_layer_weights, cur_layer_biases
            weights = weights[(m+1)*n:]

    def predictions(weights, inputs):
        for W, b in unpack_layers(weights):
            outputs = np.dot(inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(weights, inputs, targets):
        log_prior = np.sum(norm.logpdf(weights, 0, weight_scale))
        preds = predictions(weights, inputs)
        log_lik = np.sum(norm.logpdf(preds, targets, noise_scale))
        return log_prior + log_lik

    return num_weights, predictions, logprob

if __name__ == '__main__':

    # Specify inference problem by its unnormalized log-posterior.
    rbf  = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.0)

    # Implement a 3-hidden layer neural network.
    num_weights, predictions, logprob = \
        make_nn_funs(layer_sizes=[34, 100, 100, 100, 1], nonlinearity=rbf)

    ### IMPORT DATA ###
    print("IMPORTING DATA")
    X_raw, y_raw = ascdata.load_asc_data()
    X, y = ascdata.remove_zeros(X_raw, y_raw)
    X_train, y_train, X_test, y_test = ascdata.split_train_test(X, y)
    # X_bp1, y_bp1 = ascdata.get_bp_data(1, 4194659, X_train, y_train)
    # X_bp2, y_bp2 = ascdata.get_bp_data(3, 4194873, X_train, y_train)

    objective = lambda weights, t: -logprob(weights, X_train, y_train)

    # def plot_initial_data(inX, iny, ax):
    #     plot_inputs = inX.T[2]
    #     ax.plot(plot_inputs, iny.ravel(), 'bx')
    #
    # # Set up figure.
    # fig = plt.figure(1, facecolor='white')
    # ax1 = fig.add_subplot(211, frameon=False)
    # ax2 = fig.add_subplot(212, frameon=False)
    #
    # plot_initial_data(X_bp1, y_bp1, ax1)
    # plot_initial_data(X_bp2, y_bp2, ax2)

    # plt.show()

    def plot_prediction_data(inX, iny, params, ax, prog_num, bp):
        ax.cla()
        # Plot data and functions.
        plot_inputs = inX.T[2]
        ax.plot(plot_inputs, iny.ravel(), 'bx')
        outputs = predictions(params, inX)
        bp_rms = ascdata.RMSE(outputs, iny)
        print("rms total for prognum", prog_num, "at breakpoint", bp, bp_rms)
        ax.plot(plot_inputs, outputs, 'g')

    LLHs = []
    LLH_xs = []

    def callback(params, t, g):
        LLH = -objective(params, t)
        LLH_xs.append(t)
        print("Iteration {} log likelihood {}".format(t, LLH))
        LLHs.append(LLH)

        y_test_pred = predictions(params, X_test)
        rms_total = ascdata.RMSE(y_test, y_test_pred)
        print("rms total", rms_total)
        plot_prediction_data(X_bp1, y_bp1, params, ax1, 1, 4194659)
        plot_prediction_data(X_bp2, y_bp2, params, ax2, 3, 4194873)
        plt.draw()
        #plt.pause(1.0/60.0)

    rs = npr.RandomState(0)
    init_params = 10 * rs.randn(num_weights)

    print("Optimizing network parameters...")
    optimized_params = adam(grad(objective), init_params,
                            step_size=0.5, num_iters=100, callback=callback)

    # fig.savefig('neuralnet_singletask_crossents.png')
    # plt.figure(2, facecolor='white')
    # plt.plot(LLH_xs, LLHs)
    # plt.savefig('neuralnet_singletask_LLH.png')