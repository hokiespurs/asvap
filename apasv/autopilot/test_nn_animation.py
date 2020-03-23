# %%
import neuralnetwork as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# set up plot
cost = []
NPLOT = 1

# initialize the NN
myNN = nn.neuralnetwork(
    3,
    1,
    [4],
    output_softmax=False,
    rand_weights_method="randpm",
    rand_biases_method="zeros",
    rand_seed=50,
)

# create data
input_data = np.array([[0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
output_data = np.array([0, 1, 1, 0])

fig, ax = plt.subplots()


def animate(i):
    activation, dadz = myNN.feed_forward_full(input_data[:, 2])
    plt.cla()
    myNN.visualize2(
        ax=ax,
        neuron_face_color_param=activation,
        neuron_size_param=40,
        weight_color_param=myNN.weights,
        weight_size_param=myNN.weights,
    )
    myNN.train(
        input_data,
        output_data,
        num_iter=NPLOT,
        num_subsample_inputs=4,
        learning_rate=2,
    )


anim = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=False)

anim.save("animation.mp4")

print("done")
