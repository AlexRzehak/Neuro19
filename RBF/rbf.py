"""
Group: ---> (Alexander Rzehak, Aydeniz Soezbilir) <---
"""

import numpy as np
import matplotlib.pyplot as plt


STANDARD_RANGE = (-5, 5)
STANDARD_NOISE = (-0.1, 0.1)


def periodic_function(x):
    """Task a): Implement y = sin(x) + 0.5"""
    return np.sin(x) + 0.5 * x


def generate_training_examples(number, x_range: tuple = STANDARD_RANGE,
                               noise: tuple = STANDARD_NOISE):
    # get n random values for x
    x_vec = np.random.uniform(*x_range, number)
    # calcualate f(x)
    y_vec = periodic_function(x_vec)
    # add noise
    noise_vec = np.random.uniform(*noise, number)
    target_vec = y_vec + noise_vec
    return (x_vec, target_vec)


def plot_training_examples(examples: tuple, x_range: tuple = STANDARD_RANGE):
    """"Task b) : Plot training examples."""
    x_acc = np.arange(*x_range, 0.01)
    plt.figure()
    plt.plot(x_acc, periodic_function(x_acc))
    plt.plot(*examples, 'ro')
    # plt.scatter(*examples, color='r')
    # plt.xlim(x_range)
    # plt.show()


def activation_function(x):
    return np.exp(-x**2/2)


def rbf_network_function(w, c, x):
    """Task c): Implement the RBF network function.
    This function will calculate the network output
    for each x-value in input vector x.
    """

    # Calcualation for one single value x.
    def netfunct_single_val(val):
        netin = val - w
        act = activation_function(netin)
        return np.matmul(c, act)

    vf = np.vectorize(netfunct_single_val)
    return vf(x)


def calculate_matrix_C(examples: tuple, weights):
    """Task d): Find values for C."""
    Xs, Y = examples
    # note, that this construction only works for one-dimensional input
    K = np.size(weights)
    N = np.size(Xs)
    Xs_mat = np.tile(Xs, [K, 1]).T
    W_mat = np.tile(weights, [N, 1])
    H_base = Xs_mat - W_mat
    # h(x-w)
    H = activation_function(H_base)
    # compose moore-penrose-pseudo-inverse
    H_plus = np.linalg.pinv(H)
    # calculate and return C = H+ * Y
    return np.matmul(H_plus, Y)


def perform_pass_for_weights(weights, examples=None,
                             x_range=STANDARD_RANGE, color='b'):
    if not examples:
        examples = generate_training_examples(10)

    C = calculate_matrix_C(examples, weights)

    # get values for the function approximated by the network
    r = np.arange(*x_range, 0.01)
    results = rbf_network_function(weights, C, r)

    # prepare plot
    plt.figure()
    plt.plot(r, results, color)


# Define weight vectors of task d)
w1 = np.array([-4, 2, -4])
w2 = np.array([-3, 3, -2])
w3 = np.array([-2, 4, 2])
w4 = np.array([-1, 5, 4])


if __name__ == "__main__":
    ex = generate_training_examples(10)
    plot_training_examples(ex)

    # insert commands for task d) here
    perform_pass_for_weights(w1, examples=ex)
    perform_pass_for_weights(w2, examples=ex)
    perform_pass_for_weights(w3, examples=ex)
    perform_pass_for_weights(w4, examples=ex)

    # using more noise for subtask g)
    ex2 = generate_training_examples(10, noise=(-0.4, 0.4))
    plot_training_examples(ex2)

    perform_pass_for_weights(w3, examples=ex2)

    # shot all plot figures
    plt.show()
