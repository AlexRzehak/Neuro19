
"""
Group: ---> (Alexander Rzehak, Aydeniz Soezbilir) <---

Your tasks:
    Fill in your names.
    Complete the methods at the marked code blocks.
    Please comment on the most important places.

run your program with
    python3 MultiLayerANN.py --key1=value1 --key2=value2 ...
e.g.
    python3 MultiLayerANN.py --epochs=1000 --inputs='xor-input.txt' --outputs='xor-target.txt' --activation='tanh'

Key/value pairs that are unchanged use their respective defaults.
"""
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt


class Sigmoid:
    @staticmethod
    def f(x: np.array) -> np.array:
        """ the sigmoid function """
        # We use python lambda expression to apply f to every array element.
        sick = lambda a: 1/(1 + np.exp(-a))
        return sick(x)
    
    @staticmethod
    def d(x: np.array) -> np.array:
        """ the first derivative """
        # We use the simple calculation of the deriavtion.
        f = Sigmoid.f(x)
        der = lambda a: a * (1 - a)
        return der(f)


class TanH:
    @staticmethod
    def f(x: np.array) -> np.array:
        """ the tanh function """
        # return np.tanh(x)
        # We use python lambda expression to apply f to every array element.
        fun = lambda a: (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
        return fun(x)

    @staticmethod
    def d(x: np.array) -> np.array:
        """ the first derivative """
        # We use the simple calculation of the deriavtion.
        f = TanH.f(x)
        der = lambda a: 1 - a ** 2
        return der(f)


class MultiLayerANN:
    """
    The class MultiLayer implements a multi layer ANN with a flexible number of hidden layers.
    Backpropagation is used to calculate the gradients.
    """

    # the activation of the Bias neuron
    BIAS_ACTIVATION = -1

    def __init__(self, act_fun, *layer_dimensions):
        """
        initializes a new MultiLayerANN object
        :param act_fun: Which activation function to use.
        :param layer_dimensions: each parameter describes the amount of neurons in the corresponding layer.
        E.g. MultiLayerANN(TanH, 3, 10, 4) creates a network with 3 layers, 3 input_ neurons, 10 hidden neurons,
            4 output neurons, and uses the tanh activation function.
        """
        if len(layer_dimensions) < 2:
            raise Exception("At least an input_ and output layer must be given")
        self._act_fun = act_fun
        self._layer_dimensions = layer_dimensions

        # the net_input value for each non input_ neuron,
        # each list element represents one layer
        self._net_inputs = []

        # Type: list of np.arrays. The activation value for each non input_ neuron,
        # each list element represents one layer
        self._activations = []
        
        # Type: list of np.arrays. The back propagation delta value for each non input_ neuron,
        # each list element represents one layer
        self._deltas = []
        
        # Type: list of np.arrays. List of all weight matrices. 
        # Weight matrices are randomly initialized between -1 and 1
        self._weights = []
        
        # Type: list of np.arrays. List of all delta weight matrices. 
        # They are added to the corresponding weight matrices after each training step
        self._weights_deltas = []

        for i in range(len(self._layer_dimensions[1:])):
            layer_size, prev_layer_size = self._layer_dimensions[i+1], self._layer_dimensions[i]
            self._net_inputs.append(np.zeros(layer_size))
            self._activations.append(np.zeros(layer_size))
            self._deltas.append(np.zeros(layer_size))

            # we use +1 to consider the bias-neurons
            # weights are chosen randomly (uniform distribution) between -1 and 1
            self._weights.append(np.random.rand(prev_layer_size+1, layer_size) * 2 - 1)
            self._weights_deltas.append(np.zeros([prev_layer_size + 1, layer_size]))

    def _predict(self, input_: np.array) -> np.array:
        """
        calculates the output of the network for one input vector
        :param input_: input vector
        :return: activation of the output layer
        """

        # Since we don't want code duplication, we don't create
        # an exception for the first layer after the input layer.
        # Therefore, we need to modify our activations array to
        # contain the output of the input neurons.
        # Remember to change it back after!
        self._activations.insert(0, input_)

        # Now iterate over every layer:
        for i in range(len(self._layer_dimensions[1:])):
            # We need to add the input of the Bias neuron to the activations.
            activations_bias = np.append(self._activations[i], MultiLayerANN.BIAS_ACTIVATION)
            # Calculate the network input if layer i
            # from the output of layer i-1 and the weight matrix w_i-1,i
            self._net_inputs[i] = np.matmul(activations_bias, self._weights[i])
            # Calculate the activation values of layer i
            # from network input and activation function.
            # Put it in the right place in the activation array
            # for the next iteration to process.
            self._activations[i+1] = self._act_fun.f(self._net_inputs[i])

        # Don't forget to cut the input values inserted above.
        self._activations = self._activations[1:]

        # The activations of the last layer are the output of the network.            
        return self._activations[-1]

    def _train_pattern(self, input_: np.array, target: np.array, lr: float, momentum: float, decay: float):
        """
        trains one input vector
        :param input_: one input vector
        :param target: one target vector
        :param lr: learning rate
        :param momentum:
        :param decay: weight decay:
        """
        # Perform forward pass to get acces to the current activation values.
        self._predict(input_)

        # compute mean squared error
        d = (target - self._activations[-1])
        error = np.dot(d.T, d) / (2 * len(d))
        # since the program re-calculates these values at another place,
        # for now it won't make use of this.

        # - first compute output layer deltas
        der = self._act_fun.d(self._net_inputs[-1])
        dif = target - self._activations[-1]
        self._deltas[-1] = np.multiply(der, dif)

        # - then compute hidden layer deltas, consider that no delta is needed for the bias neuron
        # iterate over layers in reverse order: we don't need deltas for input
        # neurons and have the values for the output neurons calculated above.
        max_layer = len(self._layer_dimensions) - 2
        for i in reversed(range(max_layer)):
            der = self._act_fun.d(self._net_inputs[i])
            # we don't need to calculate a delta for the bias neuron,
            # therefore we cut the corresponding edges out of the matrix
            weights = self._weights[i + 1][0 : -1]
            # calculate the deltas of layer i from the deltas
            # of layer i + 1 and the weights w_i,i+1
            vals = np.matmul(weights, self._deltas[i + 1])
            self._deltas[i] = np.multiply(der, vals)

        #   Note: self._deltas[0:-1] = ignore last delta
        # for delta, last_delta, weights, net_inputs in zip(reversed(self._deltas[0:-1]), reversed(self._deltas[1:]),
        #                                                   reversed(self._weights[1:]), reversed(self._net_inputs[0:-1])):
        #     pass
        # This implementation can't change the values at the array positions without knowing their index.

        # compute weight updates:
        # add input layer activations to activations and ignore output layer activations
        act_with_input = [input_] + self._activations[0:-1]
        for i in range(len(self._layer_dimensions[1:])):
            # again, add the activation from the bias neuron
            # to the activations of layer i-1
            activations_bias = np.append(act_with_input[i], MultiLayerANN.BIAS_ACTIVATION)
            # calculate the weight changes as shown in task 2
            vals = np.outer(activations_bias, self._deltas[i])
            self._weights_deltas[i] = lr * vals
            # add the deltas to the weights of the current level.
            self._weights[i] = self._weights[i] + self._weights_deltas[i]

        # for weights_layer, weight_deltas_layer, activation_layer, delta_layer in zip(self._weights, self._weights_deltas, act_with_input, self._deltas):
        #     pass
        # Same goes here:
        # This implementation can't change the values at the array positions without knowing their index.

    def train(self, inputs: [np.array], targets: [np.array], epochs: int, lr: float, momentum: float, decay: float) -> list:
        """
        trains a set of input_ vectors. The error for each epoch gets printed out.
        In addition, the amount of correctly classiefied input_ vectors in printed
        :param inputs: list of input_ vectors
        :param targets: list of target vectors
        :param epochs: number of training iterations
        :param lr: learning rate for the weight update
        :param momentum: momentum for the weight update
        :param decay: weight decay for the weight update
        :return: list of errors. One error value for each epoch. One error is the mean error over all input_vectors
        """
        errors = []
        for epoch in range(epochs):
            error = 0
            for input_, target in zip(inputs,targets):
                self._train_pattern(input_, target, lr, momentum, decay)
            for input_, target in zip(inputs, targets):
                output = self._predict(input_)
                d = (target-output)
                error += np.dot(d.T, d) / len(d)
            error /= len(inputs)
            errors.append(error)
            print("epoch: {0:d}   error: {1:f}".format(epoch, float(error)))

        print("final error: {0:f}".format(float(errors[-1])))

        # evaluate the prediction
        correct_predictions = 0
        for input_, target in zip(inputs, targets):
            # for one output use a threshold with 0.5
            if isinstance(target, float):
                correct_predictions += 1 if np.abs(self._predict(input_)-target) < 0.5 else 0

            # for multiple outputs choose the outputs with the highest value as predicted class
            else:
                prediction = self._predict(input_)
                predicted_class = np.argmax(prediction)
                correct_predictions += 1 if target[predicted_class] == 1 else 0
        print("correctly classified: {0:d} / {1:d}".format(correct_predictions, len(inputs)))
        return errors


def read_double_array(filename):
    """
    reads an np.array from the provided file.
    :param filename: path to a file
    :return: np.array of the matrix given in the file
    """
    with open(filename) as file:
        content = file.readlines()
        return np.loadtxt(content)


def main():
    parser = argparse.ArgumentParser("MultiLayerANN")
    parser.add_argument('--inputs', type=str, default='digits-input.txt')
    parser.add_argument('--outputs', type=str, default='digits-target.txt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
    parser.add_argument('--decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--activation', type=str, default='sigmoid', help='sigmoid or tanh')
    args = parser.parse_args()

    # or be lazy, don't pass runtime args and change them here, e.g.
    # v = 'xor-input.txt', 'xor-target.txt', 100, 0.1, 0.0, 0.0, 'tanh'
    # args.inputs, args.outputs, args.epochs, args.lr, args.momentum, args.decay, args.activation = v
    print(args)

    input_vectors = read_double_array(args.inputs)
    targets = read_double_array(args.outputs)
    fun = Sigmoid if args.activation.lower() == 'sigmoid' else TanH
    num_outputs = 1 if isinstance(targets[0], float) else targets.shape[1]

    # Here, we can changethe layout of the network.
    net = MultiLayerANN(fun, input_vectors.shape[1], 15, 15, num_outputs)
    errors = net.train(input_vectors, targets, args.epochs, args.lr, args.momentum, args.decay)

    plt.plot(errors, 'r')
    plt.ylabel('error')
    plt.xlabel('iteration')
    plt.show(block=True)


if __name__ == "__main__":
    main()





