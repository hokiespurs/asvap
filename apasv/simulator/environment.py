import numpy as np

# from scipy.interpolate import RegularGridInterpolator
# from scipy.misc import imread


class environment:
    def __init__(
        self,
        axis_lims=[-100, 100, -100, 100],
        currents_data=None,
        depth_data=None,
        background_image=None,
        current_x_scale=1,
        current_y_scale=1,
        depth_scale=1,
    ):
        """
        Class to hold environment parameters (currents, depths, background image)
        """
        # TODO: decide on how to encode/read vector fields and depths
        if currents_data is None:
            self.get_currents = self.default_currents
        elif type(currents_data) is str:
            self.get_currents = self.__read_currents(
                currents_data, current_x_scale, current_y_scale, axis_lims
            )
        else:
            x_current = current_x_scale * currents_data[0]
            y_current = current_y_scale * currents_data[1]
            self.get_currents = lambda xy: [x_current, y_current]
            # pass

        # return function to get depth data f(xy)
        if depth_data is None:
            # self.get_depths = lambda xy: 0
            pass
        elif type(depth_data) is str:
            self.get_depths = self.__read_depths(depth_data, depth_scale, axis_lims)
        else:
            pass
            # depth_vals = depth_data[0] * depth_scale
            # self.get_depths = lambda xy: depth_vals

        # stores background image
        self.axis_lims = axis_lims
        if background_image is None:
            self.background_image = None
        else:
            self.background_image = None
            # self.background_image = imread(background_image)

    @staticmethod
    def default_currents(xy):
        return (0, 0)

    @staticmethod
    def __read_currents(fname, current_x_scale, current_y_scale, axis_lims):
        if fname == "test":
            return currents_test
        if fname == "line":
            return currents_straight_out
        else:
            return lambda xy: (0, 0)

    @staticmethod
    def __read_depths(fname, depth_scale, axis_lims):
        # TODO implement reading of depths
        pass
        # return lambda xy: [0]


def currents_test(xy):
    if xy[1] < 0:
        # strong currents increasing to 3m/s horizontal at -100
        x_current = xy[1] / 100 * 1
        return (x_current, 0)
    elif xy[1] < 20:
        return (0, 0)
    elif xy[1] < 40:
        return (0.1, 0)
    elif xy[1] < 60:
        return (0, -0.1)
    elif xy[1] < 80:
        return (-0.2, 0.2)
    elif xy[1] < 100:
        return (-0.25, -0.3)
    elif xy[0] < 20:
        return (0, 0.2)
    elif xy[0] < 60:
        return (0.1, 0)
    return (0, 0.2)


def currents_straight_out(xy):
    current_data_x = [
        {"start_pos": 50, "end_pos": 100, "run_in": 25, "run_out": 25, "mag": 0.1},
        {"start_pos": 125, "end_pos": 175, "run_in": 20, "run_out": 20, "mag": -0.15},
        {"start_pos": 200, "end_pos": 250, "run_in": 15, "run_out": 15, "mag": 0.2},
        {"start_pos": 260, "end_pos": 300, "run_in": 20, "run_out": 20, "mag": 0.4},
        {"start_pos": 300, "end_pos": 375, "run_in": 25, "run_out": 25, "mag": -0.5},
        {"start_pos": 400, "end_pos": 450, "run_in": 15, "run_out": 5, "mag": 1},
        {"start_pos": 450, "end_pos": 500, "run_in": 3, "run_out": 3, "mag": -1},
        {"start_pos": 450, "end_pos": 500, "run_in": 3, "run_out": 3, "mag": -1.25},
        {"start_pos": 500, "end_pos": 600, "run_in": 3, "run_out": 3, "mag": 0.1},
    ]
    current_data_y = [
        {"start_pos": 25, "end_pos": 100, "run_in": 25, "run_out": 25, "mag": -0.1},
        {"start_pos": 125, "end_pos": 175, "run_in": 20, "run_out": 20, "mag": 0},
        {"start_pos": 200, "end_pos": 250, "run_in": 15, "run_out": 15, "mag": 0},
        {"start_pos": 260, "end_pos": 300, "run_in": 20, "run_out": 20, "mag": 0},
        {"start_pos": 300, "end_pos": 375, "run_in": 25, "run_out": 25, "mag": 0.2},
        {"start_pos": 400, "end_pos": 450, "run_in": 15, "run_out": 5, "mag": 0.1},
        {"start_pos": 450, "end_pos": 500, "run_in": 3, "run_out": 3, "mag": 0},
        {"start_pos": 450, "end_pos": 500, "run_in": 3, "run_out": 3, "mag": 0},
        {"start_pos": 500, "end_pos": 550, "run_in": 3, "run_out": 3, "mag": -1},
        {"start_pos": 550, "end_pos": 600, "run_in": 3, "run_out": 3, "mag": 1},
    ]
    x_current = sin_currents(xy[1], current_data_x)
    y_current = sin_currents(xy[1], current_data_y)
    return [x_current, y_current]


def sin_currents(x, all_data):
    # list of [start_pos, end_pos, run_in, run_out]

    for dat in all_data:

        if x >= dat["start_pos"] and x < dat["start_pos"] + dat["run_in"]:
            y = np.cos((x - dat["start_pos"]) * 2 * np.pi / (dat["run_in"] * 2))
            return -(y - 1) / 2 * dat["mag"]
        elif (
            x >= dat["start_pos"] + dat["run_in"]
            and x < dat["end_pos"] - dat["run_out"]
        ):
            return dat["mag"]
        elif x >= dat["end_pos"] - dat["run_out"] and x < dat["end_pos"]:
            y = np.cos((dat["end_pos"] - x) * 2 * np.pi / (dat["run_out"] * 2))
            return -(y - 1) / 2 * dat["mag"]

    return 0


if __name__ == "__main__":
    current_data = [
        {"start_pos": 0, "end_pos": 50, "run_in": 20, "run_out": 20, "mag": 1},
        {"start_pos": 50, "end_pos": 70, "run_in": 5, "run_out": 3, "mag": 2},
        {"start_pos": 70, "end_pos": 80, "run_in": 2, "run_out": 4, "mag": 3},
    ]

    all_x = np.arange(0, 100, 0.5)
    y = [sin_currents(x, current_data) for x in all_x]
    import matplotlib.pyplot as plt

    plt.plot(all_x, y)
    plt.show()
