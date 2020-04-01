# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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
        num_input,
        num_output,
        num_neurons,
        output_softmax=False,
        activation_function_names=None,  # "sigmoid", "relu", "tanh"
        weights=None,
        biases=None,
        rand_seed=14,
        rand_weights_method="rand",  # "rand","randn","randpm"
        rand_weights_scalar=1,
        rand_biases_method="rand",  # "rand","randn","randpm","zero"
        rand_biases_scalar=1,
    ):
        self.num_input = num_input
        self.num_output = num_output
        self.num_neurons = num_neurons
        self.output_softmax = output_softmax
        self.rand_seed = rand_seed
        self.random_generator = np.random.default_rng(rand_seed)
        self.rand_weights_method = rand_weights_method
        self.rand_weights_scalar = rand_weights_scalar
        self.rand_biases_method = rand_biases_method
        self.rand_biases_scalar = rand_biases_scalar

        # if no activation_functions input, use sigmoid for all layers
        if activation_function_names is None:
            activation_function_names = ["sigmoid"] * len(num_neurons) + ["sigmoid"]

        self.activation_functions = self.get_function_handles(activation_function_names)

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
        self.check_nn_structure()

    def random_biases(self):
        """ Initialize random biases """
        # use rand_seed+1 so values arent same as weights
        _, correct_bias_shapes = self.get_correct_nn_sizes()

        self.biases = []
        for bias_shape in correct_bias_shapes:
            randvals = self.get_random_vals(
                self.rand_biases_method,
                bias_shape[0],
                bias_shape[1],
                self.rand_biases_scalar,
            )
            self.biases.append(randvals)

    def random_weights(self):
        """ Initialize random weights """
        correct_weight_shapes, _ = self.get_correct_nn_sizes()

        self.weights = []
        for weight_shape in correct_weight_shapes:
            randvals = self.get_random_vals(
                self.rand_weights_method,
                weight_shape[0],
                weight_shape[1],
                self.rand_weights_scalar,
            )
            self.weights.append(randvals)

    def get_random_vals(self, method, num_row, num_col, scalar=1):
        """ return random sample from different population methods """
        if method == "rand":
            return self.random_generator.random((num_row, num_col)) * scalar
        elif method == "randn":
            return self.random_generator.standard_normal((num_row, num_col)) * scalar
        elif method == "zeros":
            return np.zeros((num_row, num_col))
        elif method == "randpm":
            return (2 * self.random_generator.random((num_row, num_col)) - 1) * scalar
        else:
            # this exception should be raised in the weights/bias functions beforehand
            raise ValueError("Unknown method ('rand','randn','randpm','zeros')")

    def feed_forward(self, input_data):
        """ Feed forward through the neural network """
        # check shape
        input_data = self.check_input(input_data)

        # feed forward
        #   f(weights*activation + bias)
        #   f([nxm]*[mx1] + [nx1])
        activation = input_data.copy()
        for weight, bias, fun in zip(
            self.weights, self.biases, self.activation_functions
        ):
            activation = fun(weight.dot(activation) + bias)

        return activation

    def feed_forward_full(self, input_data):
        """ Feed forward through the neural network recording all data """
        # check shape
        input_data = self.check_input(input_data)

        # feed forward
        #   f(weights*activation + bias)
        #   f([nxm]*[mx1] + [nx1])
        activation = [input_data.copy()]
        dadz = []
        for weight, bias, fun in zip(
            self.weights, self.biases, self.activation_functions
        ):
            z = weight.dot(activation[-1]) + bias
            dadz.append(fun(z, True))
            activation.append(fun(z))

        return activation, dadz

    def back_propagate(self, prediction_error, activation, all_dadz):
        """  Back Propagate Error Through the Neural Net """
        dC_da = 2 * prediction_error

        dw_all = []
        db_all = []

        num_observations = activation[0].shape[1]
        for w, b, a, dadz in zip(
            reversed(self.weights),
            reversed(self.biases),
            reversed(activation[0:-1]),
            reversed(all_dadz),
        ):
            dC_dw = np.dot(dadz * dC_da, a.T) / num_observations
            dC_db = 1.0 * dadz * dC_da
            dC_da = w.T.dot(dadz * dC_da)

            # take average of gradients
            dw_all.append(dC_dw)
            db_all.append(np.mean(dC_db, axis=1).reshape(-1, 1))

        # Return mean of partial derivatives
        dw_all.reverse()
        db_all.reverse()
        return dw_all, db_all, dC_da

    def train(
        self,
        input_data,
        output_data,
        num_iter=10000,
        learning_rate=0.1,
        do_print_status=True,
        num_subsample_inputs=None,
        rand_seed=1,
    ):
        """ Train the neural network weights and biases"""
        # check shapes
        output_data = self.check_output(output_data)
        input_data = self.check_input(input_data)
        n_observations = input_data.shape[1]

        # set random seed
        if num_subsample_inputs is not None:
            np.random.seed(rand_seed)

        mean_cost = np.zeros((num_iter, 1))
        for i in range(num_iter):
            # get a subset of the data to train on
            if num_subsample_inputs is not None:
                ind = np.random.choice(
                    n_observations, size=num_subsample_inputs, replace=False
                )
                training_inputs = input_data[:, ind].reshape(self.num_input, -1)
                training_outputs = output_data[:, ind].reshape(self.num_output, -1)
            else:
                training_inputs = input_data
                training_outputs = output_data
            # forward prop
            activation, dadz = self.feed_forward_full(training_inputs)
            # compute error
            prediction_error = activation[-1] - training_outputs
            # compute total cost for stats
            mean_cost[i] = np.mean(prediction_error ** 2)
            # backpropogate to get partial derivatives
            dCdw, dCdb, dCda = self.back_propagate(prediction_error, activation, dadz)
            # update weights/biases
            for dw, db, w, b in zip(dCdw, dCdb, self.weights, self.biases):
                w -= dw * learning_rate
                b -= db * learning_rate
        return mean_cost

    def get_function_handles(self, function_names):
        """ returns list of function handles for each layer"""
        # "sigmoid", "relu", "tanh", "softmax"
        function_handles = []
        for i, fname in enumerate(function_names):
            i_function_handle = get_function_handle_by_name(fname)
            if i == len(function_names) - 1 and self.output_softmax:  # output layer
                function_handles.append(
                    lambda x, d=False: softmax(x, i_function_handle, d)
                )
            else:
                function_handles.append(i_function_handle)

        return function_handles

    def get_nn_vector(self):
        """ return weights and biases as a vectr """
        # creates [m x 1] vector [w1[:] w2[:] ... wn[:]  b1[:] b2[:] ... bn[:]]
        weight_vector = np.concatenate(self.weights, axis=None)
        bias_vector = np.concatenate(self.biases, axis=None)
        return np.concatenate([weight_vector, bias_vector], axis=None)

    def set_nn_vector(self, nn_vector):
        """ set the weights and biases based on the input nn_vector"""
        new_weights = []
        new_biases = []
        cur_ind = 0  # running flag for current index

        # get weights from vector
        for i_weight in self.weights:
            (weight_rows, weight_cols) = i_weight.shape
            end_ind = cur_ind + weight_rows * weight_cols
            wval = nn_vector[cur_ind:end_ind].reshape(weight_rows, weight_cols)
            cur_ind = end_ind
            new_weights.append(wval)

        # get biases from vector
        for i_bias in self.biases:
            bias_rows = i_bias.size
            end_ind = cur_ind + bias_rows
            bval = nn_vector[cur_ind:end_ind].reshape(bias_rows, 1)
            cur_ind = end_ind

            new_biases.append(bval)

        self.weights = new_weights
        self.biases = new_biases

    def get_correct_nn_sizes(self):
        """ computes correct shape of numpy array in weight and bias lists """
        num_neurons = np.concatenate(
            [self.num_input, self.num_neurons, self.num_output], axis=None
        )
        correct_weight_shapes = []
        correct_bias_shapes = []
        for num_input_neurons, num_output_neurons in zip(
            num_neurons[0:-1], num_neurons[1:]
        ):
            correct_weight_shapes.append((num_output_neurons, num_input_neurons))
            correct_bias_shapes.append((num_output_neurons, 1))
        return (correct_weight_shapes, correct_bias_shapes)

    def check_input(self, input_data):
        """ throw error if shape of input_data is wrong"""
        input_data = np.array(input_data)
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(self.num_input, -1)
        if input_data.shape[0] != self.num_input:
            raise ValueError(
                f"Input data [{input_data.shape}] =/= ({self.num_input} x N)"
            )

        return input_data

    def check_output(self, output_data):
        """ throw error if shape of output_data is wrong"""
        output_data = np.array(output_data)
        if len(output_data.shape) == 1:
            output_data = output_data.reshape(self.num_output, -1)

        if output_data.shape[0] != self.num_output:
            raise ValueError(
                f"output data [{output_data.shape}] =/= [{self.num_output} x N]"
            )

        return output_data

    def check_nn_structure(self):
        """ throw error if shape of weights or biases are wrong"""
        # inputs  = [n1 x 1]
        # weights = [n2 x n1]
        # biases  = [n2 x 1]
        (correct_weight_shapes, correct_bias_shapes) = self.get_correct_nn_sizes()
        for true_weight_shape, true_bias_shape, layer_weight, layer_bias in zip(
            correct_weight_shapes, correct_bias_shapes, self.weights, self.biases
        ):

            if layer_bias.size == 1:
                layer_bias = layer_bias.reshape(1, 1)
            if layer_weight.size == 1:
                layer_weight = layer_weight.reshape(1, 1)

            if layer_weight.shape != true_weight_shape:
                raise ValueError(
                    f"Weight matrix [{layer_weight.shape}] =/= [{true_weight_shape}]"
                )
            if layer_bias.shape != true_bias_shape:
                errorstr = f"Bias matrix [{layer_bias.shape}] =/= [{true_bias_shape}]"
                raise ValueError(errorstr)
        # check list of activation_functions
        num_functions = len(self.activation_functions)
        num_neurons = np.concatenate(
            [self.num_input, self.num_neurons, self.num_output], axis=None
        )

        if num_functions != len(num_neurons) - 1:
            errorstr = f"[# Functions :{num_functions}]  =/= [{len(num_neurons)-1}]"
            raise ValueError(errorstr)

        for i, fun in enumerate(self.activation_functions):
            x = np.array(0)
            try:
                fun(x)
                fun(x, True)
            except Exception:
                raise Exception("Something is wrong with function[{i}]")

    def visualize2(
        self,
        ax=None,
        neuron_face_color_param="w",
        neuron_edge_color_param="k",
        weight_color_param="b",
        neuron_size_param=40,
        weight_size_param=1,
        neuron_face_cmap_name="seismic",
        neuron_edge_cmap_name="seismic",
        weight_color_cmap_name="seismic",
        neuron_face_cmap_num=11,
        neuron_edge_cmap_num=11,
        weight_color_cmap_num=11,
        weight_size_range=[0.1, 5],
        neuron_size_range=[10, 80],
        neuron_face_color_param_range=[-1, 1],
        neuron_edge_color_param_range=[-1, 1],
        weight_color_param_range=[-1, 1],
        weight_size_param_range=[-1, 1],
        neuron_size_param_range=[-1, 1],
    ):
        """ Plot neurons, weights in different sizes and colors """
        # make colormap function handles
        cmap_neuron_face = self.make_visualize_cmap(
            neuron_face_color_param,
            neuron_face_color_param_range,
            neuron_face_cmap_name,
            neuron_face_cmap_num,
        )
        cmap_neuron_edge = self.make_visualize_cmap(
            neuron_edge_color_param,
            neuron_edge_color_param_range,
            neuron_edge_cmap_name,
            neuron_edge_cmap_num,
        )
        cmap_weight = self.make_visualize_cmap(
            weight_color_param,
            weight_color_param_range,
            weight_color_cmap_name,
            weight_color_cmap_num,
        )

        # make size caluculation function handles
        size_neuron = self.make_visualize_size(
            neuron_size_param, neuron_size_range, neuron_size_param_range
        )
        size_weight = self.make_visualize_size(
            weight_size_param, weight_size_range, weight_size_param_range
        )

        # calculate spacing
        num_neurons = np.concatenate(
            [self.num_input, self.num_neurons, self.num_output], axis=None
        )
        x_dim = np.linspace(1, len(num_neurons), len(num_neurons))
        x_dim /= np.mean(x_dim)
        y_dim = []
        for i_nuerons in num_neurons:
            y_vals = np.linspace(1, i_nuerons, i_nuerons)
            y_vals -= np.mean(y_vals)
            y_dim.append(y_vals)

        # make axes handle if it isn't input
        do_plt_show = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            do_plt_show = True

        # plot and color weights
        for layer_num, w in enumerate(self.weights):
            for i_row in range(w.shape[0]):
                for j_col in range(w.shape[1]):
                    # get line color value
                    if type(weight_color_param) is str:
                        line_color_val = weight_color_param
                    else:
                        line_color_val = weight_color_param[layer_num][i_row][j_col]

                    # get face size value
                    if type(weight_size_param) is int:
                        line_size_val = weight_size_param
                    else:
                        line_size_val = weight_size_param[layer_num][i_row][j_col]

                    x_plot = [x_dim[layer_num + 1], x_dim[layer_num]]
                    y_plot = [y_dim[layer_num + 1][i_row], y_dim[layer_num][j_col]]
                    ax.plot(
                        x_plot,
                        y_plot,
                        color=cmap_weight(line_color_val),
                        linewidth=size_weight(line_size_val),
                    )
        # plot neurons
        for layer_num, b in enumerate(self.biases):
            for i in range(b.size):
                # get face color value
                if type(neuron_face_color_param) is str:
                    face_color_val = neuron_face_color_param
                else:
                    if len(neuron_face_color_param) == len(self.biases):
                        face_color_val = neuron_face_color_param[layer_num][i][0]
                    else:
                        face_color_val = neuron_face_color_param[layer_num + 1][i][0]

                # get edge color value
                if type(neuron_edge_color_param) is str:
                    edge_color_val = neuron_edge_color_param
                else:
                    if len(neuron_face_color_param) == len(self.biases):
                        edge_color_val = neuron_edge_color_param[layer_num][i]
                    else:
                        edge_color_val = neuron_edge_color_param[layer_num + 1][i]

                # get face size value
                if type(neuron_size_param) is int:
                    size_val = neuron_size_param
                else:
                    if len(neuron_size_param) == len(self.biases):
                        size_val = neuron_size_param[layer_num][i]
                    else:
                        size_val = neuron_size_param[layer_num + 1][i]

                ax.plot(
                    x_dim[layer_num + 1],
                    y_dim[layer_num + 1][i],
                    ".",
                    markersize=size_neuron(size_val),
                    mec=cmap_neuron_edge(edge_color_val),
                    mfc=cmap_neuron_face(face_color_val),
                )
                # p = patches.Circle(circle_xy, radius=neuron_size / 2)
                # ax.gcf().gca().add_artist(p)

        # plot input neurons
        for i in range(y_dim[0].size):
            if type(neuron_face_color_param) is str:
                face_color_val = neuron_face_color_param
            else:
                if len(neuron_face_color_param) == len(self.biases):
                    face_color_val = "w"
                else:
                    face_color_val = neuron_face_color_param[0][i][0]

            if type(neuron_edge_color_param) is str:
                edge_color_val = neuron_edge_color_param
            else:
                if len(neuron_edge_color_param) == len(self.biases):
                    edge_color_val = "k"
                else:
                    edge_color_val = neuron_edge_color_param[0][i]

            if type(neuron_size_param) is int:
                size_val = neuron_size_param
            else:
                if len(neuron_size_param) == len(self.biases):
                    size_val = 0
                else:
                    size_val = neuron_size_param[0][i]

            ax.plot(
                x_dim[0],
                y_dim[0][i],
                ".",
                markersize=size_neuron(size_val),
                mec=cmap_neuron_edge(edge_color_val),
                mfc=cmap_neuron_face(face_color_val),
            )

        if do_plt_show:
            plt.show()

    def make_visualize_cmap(
        self, input_parameter, parameter_range, cmap_name, num_cmap=11
    ):
        """ returns a function that returns a color based on cmap and range """
        if type(input_parameter) is str:
            return lambda x: input_parameter
        else:
            cmap = cm.get_cmap(cmap_name, num_cmap)
            return lambda x: cmap(self.num_to_01_range(x, parameter_range))

    def make_visualize_size(self, input_parameter, size_range, parameter_range):
        """ returns a function that returns a size scaled based on parameter_range """
        if type(input_parameter) is not int:
            return lambda x: self.num_to_new_range(x, parameter_range, size_range)
        else:
            return lambda x: input_parameter
        pass

    def num_to_01_range(self, x, x_lims):
        return (x - x_lims[0]) / (x_lims[1] - x_lims[0])

    def num_to_new_range(self, x, x_range, new_range):
        return new_range[0] + (
            self.num_to_01_range(x, x_range) * (new_range[1] - new_range[0])
        )

    def visualize(
        self,
        input_data=None,
        x_space=1,
        y_space=1,
        neuron_size=0.5,
        line_width=5,
        cmap_neuron_name="seismic",
        cmap_weights_name="seismic",
    ):
        """ plots the neural network in a matplotlib figure """
        cmap_neurons = cm.get_cmap(cmap_neuron_name, 11)
        cmap_weights = cm.get_cmap(cmap_weights_name, 11)
        # calculate spacing
        num_neurons = np.concatenate(
            [self.num_input, self.num_neurons, self.num_output], axis=None
        )
        x_dim = np.linspace(1, len(num_neurons), len(num_neurons))
        x_dim /= np.mean(x_dim)
        y_dim = []

        nn_vector = self.get_nn_vector()
        num_weights = np.sum(num_neurons[:-1] * num_neurons[1:])
        max_weight = 1
        max_bias = np.max(np.abs(nn_vector[num_weights:]))
        if max_bias == 0:
            max_bias = np.inf

        for i_nuerons in num_neurons:
            y_vals = np.linspace(1, i_nuerons, i_nuerons)
            y_vals -= np.mean(y_vals)
            y_dim.append(y_vals)
            print(y_vals)
        print(x_dim)
        # plot and color weights
        for layer_num, w in enumerate(self.weights):
            for i_row in range(w.shape[0]):
                for j_col in range(w.shape[1]):
                    w_val = w[i_row, j_col]
                    x_plot = [x_dim[layer_num + 1], x_dim[layer_num]]
                    y_plot = [y_dim[layer_num + 1][i_row], y_dim[layer_num][j_col]]
                    plt.plot(
                        x_plot, y_plot, color=cmap_weights(0.5 + w_val / max_weight)
                    )
        # plot neurons
        for layer_num, b in enumerate(self.biases):
            for i in range(b.size):
                b_val = b[i]
                plt.plot(
                    x_dim[layer_num + 1],
                    y_dim[layer_num + 1][i],
                    ".",
                    markersize=40,
                    mec="k",
                    mfc=cmap_neurons(0.5 + b_val[0] / max_bias),
                )
                # p = patches.Circle(circle_xy, radius=neuron_size / 2)
                # plt.gcf().gca().add_artist(p)

        # plot input neurons
        for i in range(y_dim[0].size):
            # p = patches.Circle((x_dim[0], y_dim[0][i]), radius=neuron_size / 2)
            # plt.gcf().gca().add_artist(p)
            plt.plot(x_dim[0], y_dim[0][i], ".", markersize=40, mfc="w", mec="k")


def get_function_handle_by_name(function_name):
    """ Returns the Function handle"""
    if function_name == "sigmoid":
        return sigmoid
    elif function_name == "relu":
        return relu
    elif function_name == "tanh":
        return tanh
    elif function_name == "none":
        return nofun


def sigmoid(x, derivative=False):
    """ Computes sigmoid function of x """
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    """ rectified linear unit function"""
    if derivative:
        return (x > 0).astype(float)
    else:
        relu_vals = x.copy()
        relu_vals[x < 0] = 0
        return relu_vals


def tanh(x, derivative=False):
    """ hyperbolic tangent """
    if derivative:
        return 1.0 - np.tanh(x) ** 2
    else:
        return np.tanh(x)


def nofun(x, derivative=False):
    """ Just a pass-through function x = f(x) """
    if derivative:
        return np.ones_like(x)
    else:
        return x


def softmax(x, fun, derivative=False):
    """ softmax layer to normalize output layer """
    if derivative:
        return fun(x, True)
    else:
        return fun(x) / np.sum(fun(x), axis=0)


def test_ff():
    # data
    mydata = np.array([1, 2]).reshape(2, -1)
    weights = [np.array([[0, 1], [1, 0], [2, 2]]), np.array([[1, -1, 0.5]])]
    biases = [np.array([[0], [1], [-1]]), np.array([0.5])]
    # initialize NN
    myNN = neuralnetwork(
        num_input=2,
        num_output=1,
        num_neurons=[3],
        activation_function_names=["sigmoid", "sigmoid"],
        weights=weights,
        biases=biases,
    )
    # weights
    # myNN.W[0] = np.array([[0, 1], [1, 0], [2, 2]])
    # myNN.W[1] = np.array([[1, -1, 0.5]])
    # biases
    # myNN.B[0] = np.array([[0], [1], [-1]])
    # myNN.B[1] = np.array([0.5])
    # true result
    truth = sigmoid(sigmoid(2 + 0) - sigmoid(1 + 1) + sigmoid(6 - 1) / 2 + 0.5)

    myoutput = myNN.feed_forward(mydata)

    delta = truth - myoutput
    if delta < 1e-10:
        return 0
    else:
        return 1


if __name__ == "__main__":
    test_ff()

    myNN = neuralnetwork(
        3,
        2,
        [2],
        rand_seed=88,
        output_softmax=False,
        activation_function_names=["tanh", "sigmoid"],
        rand_weights_method="rand",
        rand_biases_method="zeros",
    )

    input_data = np.array([[0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
    output_data = np.array([[0, 1, 1, 0], [0, 1, 0, 1]])

    # myNN.weights[0] = np.array([[1.0, 0, 0], [0, 1, 0]]) * 10
    # myNN.weights[1] = np.array([[1.0, 0], [0, 1]]) * 10

    cost = myNN.train(
        input_data, output_data, 5000, num_subsample_inputs=2, learning_rate=1.0,
    )

    # myNN.visualize()
    # plt.show()
    activation, dadz = myNN.feed_forward_full(input_data)

    data_est = myNN.feed_forward(input_data)
    data_est_test1 = np.ones_like(data_est) * 1
    data_est_test1[0, :] = 0
    data_est_test2 = np.ones_like(data_est) * 1
    data_est_test2[1, :] = 0
    _, _, dCda1 = myNN.back_propagate(data_est_test1, activation, dadz)
    _, _, dCda2 = myNN.back_propagate(data_est_test2, activation, dadz)

    print("partial dOutput 1 wrt Inputs")
    print(dCda1 / (np.sum(np.abs(dCda1.T), axis=1).reshape(1, 4)))
    print("partial dOutput 2 wrt Inputs")
    print(dCda2 / (np.sum(np.abs(dCda2.T), axis=1).reshape(1, 4)))
    print(f"Data Estimate: {np.round(data_est,3)}")
    print(f"Data Truth   : {output_data}")

    myNN.visualize2(
        neuron_face_color_param=activation,
        neuron_size_param=40,
        weight_color_param=myNN.weights,
        weight_size_param=myNN.weights,
    )

    plt.plot(cost)
    plt.show()
