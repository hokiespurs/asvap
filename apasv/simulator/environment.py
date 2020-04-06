# import numpy as numpy
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
        else:
            return lambda xy: (0, 0)

    @staticmethod
    def __read_depths(fname, depth_scale, axis_lims):
        # TODO implement reading of depths
        pass
        # return lambda xy: [0]


def currents_test(xy):
    if xy[1] < 20:
        return (0, 0)
    elif xy[1] < 40:
        return (0.1, 0)
    elif xy[1] < 60:
        return (0, -0.1)
    elif xy[1] < 80:
        return (-0.2, 0.2)
    elif xy[1] < 100:
        return (0.25, -0.25)

    return (0, 0)
