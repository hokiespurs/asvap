import numpy as np

# lets the code call neural network from scripts outside this folder
try:
    import neuralnetwork
except ImportError:
    from autopilot import neuralnetwork


class autopilot:
    """ Base Autopilot Class  """

    def __init__(self, survey_lines=None):
        if survey_lines is None:
            survey_lines = self.make_fake_survey_line()
        self.survey_lines = survey_lines
        self.current_line = 0
        self.line_finished = True  # flag to make sure the boat goes in order
        self.mission_complete = False
        self.debug_autopilot_labels = ["AP"]
        self.debug_autopilot_label_data = 0
        self.debug_autopilot_partials = None

    def make_fake_survey_line(self):
        """ Make a fake survey line for debugging """
        lines = []
        line_dictionary = {
            "p1_x": 0,
            "p1_y": 0,
            "p2_x": 0,
            "p2_y": 200,
            "goal_speed": 1,
            "offline_std": 3,
            "speed_std": 1,
            "runup": 5,
            "runout": 5,
            "distance": 200,
            "az": 0,
        }
        lines.append(line_dictionary)
        return lines

    def update_current_line(self, pos_xy):
        line = self.survey_lines[self.current_line]

        # check if boat is past the last point, go to next point
        if self.line_finished is False:
            if self.current_line == len(self.survey_lines):
                self.mission_complete = True
                return

            downline_pos_backwards, _ = self.xy_to_downline_offline(
                pos_xy, [line["p2_x"], line["p2_y"]], [line["p1_x"], line["p1_y"]]
            )
            if downline_pos_backwards < 0:
                self.line_finished = True
                self.current_line += 1

                if self.current_line == len(self.survey_lines):
                    self.mission_complete = True
                    self.current_line = len(self.survey_lines) - 1
                    return

        line = self.survey_lines[self.current_line]
        # if a line was just finished, check if it's on the new line yet
        if self.line_finished is True:
            downline_pos, _ = self.xy_to_downline_offline(
                pos_xy, [line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]]
            )
            if downline_pos > 0:
                self.line_finished = False

    def get_pos_vel_gate(self, pos_xy, vel_xy, line_num):
        line = self.survey_lines[line_num]
        downline_pos, offline_pos = self.xy_to_downline_offline(
            pos_xy, [line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]]
        )
        downline_vel, offline_vel = self.xy_to_downline_offline(
            vel_xy, [line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]], True
        )

        return (downline_pos, offline_pos, downline_vel, offline_vel)

    @staticmethod
    def limit_throttle(throttle):
        """ convert throttle so it's limited to [-100 to 100] """
        throttle = throttle.squeeze()

        if throttle[0] > 100:
            throttle[0] = 100
        elif throttle[0] < -100:
            throttle[0] = -100

        if throttle[1] > 100:
            throttle[1] = 100
        elif throttle[1] < -100:
            throttle[1] = -100

        return throttle

    @staticmethod
    def xy_to_downline_offline(xy, p1, p2, do_vel=False):
        """ Converts only one xy point to downline, offline """
        az = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        cos_val = np.cos(az)
        sin_val = np.sin(az)
        if do_vel:
            downline = xy[0] * cos_val + xy[1] * sin_val
            offline = xy[0] * -sin_val + xy[1] * cos_val
        else:
            downline = (xy[0] - p1[0]) * cos_val + (xy[1] - p1[1]) * sin_val
            offline = (xy[0] - p1[0]) * -sin_val + (xy[1] - p1[1]) * cos_val

        return (downline, offline)


class ap_nn(autopilot):
    def __init__(
        self,
        survey_lines,
        num_neurons,
        output_softmax=False,
        activation_function_names=None,  # "sigmoid", "relu", "tanh"
        rand_seed=14,
        rand_weights_method="rand",  # "rand","randn","randpm"
        rand_weights_scalar=1,
        rand_biases_method="rand",  # "rand","randn","randpm","zero"
        rand_biases_scalar=1,
    ):
        super().__init__(survey_lines)
        self.old_data = {
            "downline_pos": 0,
            "offline_pos": 0,
            "downline_vel": 0,
            "offline_vel": 0,
            "az": 0,
            "vaz": 0,
        }
        fake_data = {"x": 0, "y": 0, "az": 0, "vx": 0, "vy": 0, "vaz": 0}
        fake_nn_input, _ = self.calc_nn_inputs(fake_data)
        self.num_input = fake_nn_input.size
        self.debug_autopilot_partials = [None] * self.num_input
        self.num_output = 2
        self.nn = neuralnetwork.neuralnetwork(
            self.num_input,
            self.num_output,
            num_neurons=num_neurons,
            output_softmax=output_softmax,
            activation_function_names=activation_function_names,
            rand_seed=rand_seed,
            rand_weights_method=rand_weights_method,
            rand_weights_scalar=rand_weights_scalar,
            rand_biases_method=rand_biases_method,
            rand_biases_scalar=rand_biases_scalar,
        )

    def calc_nn_inputs(self, new_data):
        """ calculate parameters for input to the neural network"""
        # INPUTS
        # downline_speed_error
        # offline_position
        # cos(azimuth_off_line)
        # sin(azimuth_off_line)
        # tanh(vel_offline)

        (downline_pos, offline_pos, downline_vel, offline_vel) = self.get_pos_vel_gate(
            [new_data["x"], new_data["y"]],
            [new_data["vx"], new_data["vy"]],
            self.current_line,
        )
        dist_remaining = self.survey_lines[self.current_line]["distance"] - downline_pos
        # compute velocity error
        d_speed = downline_vel - self.survey_lines[self.current_line]["goal_speed"]
        # compute azimuth off line
        boat_az = new_data["az"] * np.pi / 180
        line_az = self.survey_lines[self.current_line]["az"]
        daz = boat_az - line_az
        if daz > np.pi:
            daz -= 2 * np.pi
        elif daz < -np.pi:
            daz += 2 * np.pi

        downline_acc = downline_vel - self.old_data["downline_vel"]
        offline_acc = offline_vel - self.old_data["offline_vel"]
        az_vel = new_data["vaz"]
        az_acc = new_data["vaz"] - self.old_data["vaz"]

        # np.tanh(-x_l / 5),
        # -vx_l / 3,
        # np.tanh(-ax_l),
        # np.cos(np.deg2rad(daz)),
        # -np.sin(np.deg2rad(daz)),
        # -vaz / 90,
        # np.tanh(-aaz / 10),
        # (vy_l - 1) / 3,
        # np.tanh(ay_l),
        # 1,

        datavals = np.array(
            [
                tanh(offline_pos / 5),
                offline_vel / 3,
                tanh(offline_acc),
                np.cos(daz),
                np.sin(daz),
                az_vel / 90,
                tanh(az_acc / 10),
                tanh(d_speed / 3),
                tanh(downline_acc),
            ]
        ).reshape(-1, 1)
        labels = [
            "tanh(offline/5)",
            "offline_vel/3",
            "tanh(offline_acc)",
            "cos(az_err)",
            "sin(az_err)",
            "az_vel/90",
            "tanh(az_acc/10)",
            "tanh(vel_error/3)",
            "tanh(downline_acc)",
        ]
        flipped_input = False
        # Flip Inputs so always thinks its on positive side of line
        if offline_pos > 0:
            # Populate debug labels
            datavals = np.array(
                [
                    tanh(-offline_pos / 5),
                    -offline_vel / 3,
                    tanh(-offline_acc),
                    np.cos(-daz),
                    np.sin(-daz),
                    -az_vel / 90,
                    tanh(-az_acc / 10),
                    tanh(d_speed / 3),
                    tanh(downline_acc),
                ]
            ).reshape(-1, 1)
            labels = [
                "tanh(offline/5)",
                "offline_vel/3",
                "tanh(offline_acc)",
                "cos(az_err)",
                "sin(az_err)",
                "az_vel/90",
                "tanh(az_acc/10)",
                "tanh(vel_error/3)",
                "tanh(downline_acc)",
            ]
            flipped_input = True

        self.debug_autopilot_labels = labels
        self.debug_autopilot_label_data = datavals
        # old_data for acceleration calculations
        self.old_data = {
            "downline_pos": downline_pos,
            "offline_pos": offline_pos,
            "downline_vel": downline_vel,
            "offline_vel": offline_vel,
            "az": new_data["az"],
            "vaz": new_data["vaz"],
        }

        return datavals, flipped_input

    @staticmethod
    def calc_nn_out_to_throttle(nn_out, flipped_input):
        # standard way
        throttle = 200 * (nn_out - 0.5)

        # # [0] is throttle
        # # [1] is turn amount
        # total_fwd = 200 * (nn_out[0] - 0.5)
        # L_throttle = total_fwd + 200 * (nn_out[1] - 0.5)
        # R_throttle = total_fwd - 200 * (nn_out[1] - 0.5)
        # throttle = [L_throttle, R_throttle]
        if flipped_input:
            return np.hstack((throttle[1], throttle[0]))
        else:
            return np.hstack((throttle[0], throttle[1]))

    def calc_boat_throttle(self, new_data):
        """ calculate the throttle command to the boat"""
        # new_data = [pos_x, pos_y, pos_az, vel_x, vel_y, vel_az]
        self.update_current_line([new_data["x"], new_data["y"]])
        nn_inputs, flipped_input = self.calc_nn_inputs(new_data)
        # nn_outputs = self.nn.feed_forward(nn_inputs)
        # compute partial derivatives
        activation, dadz = self.nn.feed_forward_full(nn_inputs)
        nn_outputs = activation[-1]
        _, _, dCda1 = self.nn.back_propagate(
            np.array([1, 0]).reshape(2, 1), activation, dadz
        )
        dCda1 = dCda1 / np.max(np.abs(dCda1))

        _, _, dCda2 = self.nn.back_propagate(
            np.array([0, 1]).reshape(2, 1), activation, dadz
        )
        dCda2 = dCda2 / np.max(np.abs(dCda2))

        all_partials = []
        for i in range(self.num_input):
            all_partials.append([dCda1[i], dCda2[i]])
        all_partials.append(None)
        all_partials.append(None)
        self.debug_autopilot_partials = all_partials

        self.debug_autopilot_labels.append("NN Out Throttle L")
        self.debug_autopilot_labels.append("NN Out Throttle R")

        self.debug_autopilot_label_data = np.vstack(
            (
                self.debug_autopilot_label_data,
                nn_outputs[0] * 2 - 1,
                nn_outputs[1] * 2 - 1,
            )
        )

        throttle = self.calc_nn_out_to_throttle(nn_outputs, flipped_input)
        return self.limit_throttle(throttle)


class ap_shell(autopilot):
    def __init__(self, survey_lines, foo1=None, foo2=0):
        super().__init__(survey_lines)
        self.foo1 = foo1
        self.foo2 = foo2

    def calc_boat_throttle(self, new_data):
        """ calculate the throttle command to the boat"""
        # new_data = [pos_x, pos_y, pos_az, vel_x, vel_y, vel_az]
        # store any persistent data in class self variable
        #   - eg. accelerations, last N samples, etc
        pass


def sigmoid(x, derivative=False):
    """ Computes sigmoid function of x """
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    """ hyperbolic tangent """
    if derivative:
        return 1.0 - np.tanh(x) ** 2
    else:
        return np.tanh(x)


if __name__ == "__main__":
    my_ap = ap_nn(
        survey_lines=None,
        num_neurons=[10, 10, 10],
        output_softmax=False,
        activation_function_names=None,  # "sigmoid", "relu", "tanh"
        rand_seed=14,
        rand_weights_method="randpm",  # "rand","randn","randpm"
        rand_weights_scalar=1,
        rand_biases_method="randpm",  # "rand","randn","randpm","zero"
        rand_biases_scalar=1,
    )
    boat_data = {
        "x": 2,
        "y": 5,
        "az": -5,
        "vx": -0.1,
        "vy": 1.5,
        "vaz": 1,
    }
    print(my_ap.calc_boat_throttle(boat_data))
    print(my_ap.calc_boat_throttle(boat_data).shape)
