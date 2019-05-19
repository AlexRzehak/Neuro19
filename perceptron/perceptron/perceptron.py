"""
Group: ---> (FILL IN YOUR NAMES) <---


Your tasks:
- Fill in your names.
- Complete the methods at the marked code blocks.
- Please comment on the most important places.

Test your implementation at least with:
    perceptron.py orinput.txt oroutput.txt 1000 0.001
    perceptron.py xorinput.txt xoroutput.txt 1000 0.001
    perceptron.py test1input.txt test1output.txt 1000 0.001

Have fun!
"""

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt


class Perceptron:
    """
    The class Perceptron implements a single-layer perceptron
    with an arbitrary input dimension and binary step function
    as activation. The output is either 1 or 0
    """

    def __init__(self, dimension):
        """
        initializes a new Perceptron object
        :param dimension (int): dimension of the input data
        """
        self._dimension = dimension

        # random initialization of weights between -0.1 and 0.1:
        
        # TODO
        self._weights = np.matrix(np.random.uniform(-0.1, 0.1, (self._dimension, 1)))
        # self._weights = np.zeros([self._dimension])

        # Bias that has to be added to the activation. It is to the negative of the bias introduced in chapter 6.2
        # It can, but doesn't have to be learned to solve this task.
        self._bias_weight = -1.0

    def predict(self, input_: np.matrix) -> int:
        """
        Calculates the output of the perceptron for an input vector
        :param input_: input vector
        :return: network output (1 or 0)
        """
        # TODO
        network_input = input_ * self._weights + self._bias_weight

        return np.heaviside(network_input, 1)

    def _train_pattern(self, input_: np.matrix, target: int, lr: float):
        """
        If the output does not correspond to the desired value,
        the weights and bias are changed according to the delta rule.

        :param input: input vector
        :param target: desired output value (1 or 0)
        :param lr: the learning rate of the percepton learning rule
        """
        # TODO change bias
        prediction = self.predict(input_)
        deltas = lr * input_ * (target - prediction)
        self._weights = self._weights + deltas
        pass

    def train(self, input_vectors: np.matrix, targets: np.array, iterations: int, lr: float) -> list:
        """
        Trains the perceptron for a set of input vector over a specified number of iterations.
        Returns the number of incorrectly classified training vectors.
        :param input_vectors: two dimensional matrix with training vectors as rows.
                input_vectors[0,:] is the first training vector, input_vectors[1,:] the second...
        :param targets: target outputs. targets[0] is the  desired output
                for the first training vector.
        :param iterations: Number of iterations. One iteration corresponds to
                a unique call of the _train_pattern method for each training vector.
        :param lr:
        :return: average error ratio for every iteration
        """

        # TODO
        return [0]*iterations


def read_double_array(filename) -> np.array:
    """
    reads data from the provided files.
    :param filename: path to a file
    :return: np.array of the matrix given in the file
    """
    with open(filename) as file:
        content = file.readlines()
        # remove first line
        content = content[1:]
        return np.loadtxt(content)


def read_double_matrix(filename):
    """
    reads data from the provided files.
    :param filename: path to a file
    :return: np.matrix of the matrix given in the file
    """
    return np.matrix(read_double_array(filename))


def main():
    parser = argparse.ArgumentParser("perceptron")
    parser.add_argument('inputs', type=str)
    parser.add_argument('outputs', type=str)
    parser.add_argument('iterations', type=int)
    parser.add_argument('lr', type=float)
    args = parser.parse_args()

    print('arguments:', args)
    input_vectors = read_double_matrix(args.inputs)
    targets = read_double_array(args.outputs)
    perceptron = Perceptron(np.size(input_vectors[0]))

    errors = perceptron.train(input_vectors, targets, args.iterations, args.lr)


if __name__ == "__main__":
    main()