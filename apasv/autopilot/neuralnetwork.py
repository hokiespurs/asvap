import numpy as np


class neuralnetwork:
    """
    Neural network with:
        input  [num_inputs x num_samples]
        output [num_outputs x num_samples]
        weights [[num_inputs x num_neurons1],[num_neurons1,num_neurons2],...]
        biases [[1 x num_neurons1],[1 x num_neurons2],...]
    """

    def __init__(
        self,
        num_input=2,
        num_output=3,
        num_neurons=[10, 10],
        output_function="softmax",
        activation_function_names=None,
        weights=None,
        biases=None,
        rand_seed=14,
    ):
        self.num_input = num_input
        self.num_output = num_output
        self.num_neurons = num_neurons
        self.output_function = output_function
        self.rand_seed = 14
        # if no activation_functions input, use sigmoid for all layers
        if activation_function_names is None:
            activation_function_names = np.repeat("sigmoid", len(num_neurons))

        self.activation_functions = self.set_function_handles(activation_function_names)

        # if no weights or no biases, set random seed number
        if weights is None or biases is None:
            np.random.seed(self.rand_seed)

        # if no weights input, populate with random values
        if weights is None:
            self.random_weights()
        else:
            self.weights = weights

        # if no biases input, populate with random values
        if biases is None:
            self.random_biases()
        else:
            self.biases = biases

        # make sure weights and biases are formatted correctly
        self.check_weights_and_biases()

    def random_biases(self):
        """ Initialize random biases """
        pass

    def random_weights(self):
        """ Initialize random weights """
        pass

    def feed_forward(self, input_data):
        """ Feed forward the neural network """
        # check shape

        pass

    def back_propagate(self, output_errors):
        """
        Back Propagate Error Through the Neural Net
        """
        # check shape

        # Return Partial Derivatives or mean of partial drivatives

        pass

    def train(
        self,
        input_data,
        output_data,
        num_iter=10000,
        num_costs_to_return=100,
        do_print_status=True,
        num_subsample_inputs=0,
    ):
        """ Train the neural network weights and biases"""
        # check shapes
        # loop
        # forward prop
        # calculate errors
        # if cost_calc then compute total cost
        # back prop errors
        # update weights/biases

        pass

    def get_nn_vector(self):
        """ return weights and biases as a vectr """
        # creates [m x 1] vector [w1[:] b1[:] w2[:] b2[:] ... wn[:] bn[:]]
        nn_vec = np.array([]).reshape(-1, 1)
        for i_weights, i_biases in zip(self.weights, self.biases):
            nn_vec = np.vstack((nn_vec, i_weights.reshape(-1, 1)))
            nn_vec = np.vstack((nn_vec, i_biases.reshape(-1, 1)))
        return nn_vec

    def set_nn_vector(self, nn_vector):
        """ set the weights and biases based on the input nn_vector"""
        pass

    def check_input_shape(self, input_data):
        """ throw error if shape of input_data is wrong"""
        pass

    def check_output_shape(self, output_data):
        """ throw error if shape of output_data is wrong"""
        pass

    def check_weights_and_biases(self):
        """ throw error if shape of weights or biases are wrong"""
        pass


def sigmoid(x):
    """ Computes sigmoid function of x """
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    """ Derivative of sigmoid function """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    """ rectified linear unit function """
    relu_vals = x.copy()
    relu_vals[x < 0] = 0
    return relu_vals


def d_relu(x):
    """ Derivative of rectified linear unit function """
    # dx = 0 when x<0, dx=1 when x>0
    return (x > 0).astype(float)


if __name__ == "__main__":
    pass
