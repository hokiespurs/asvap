# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from time import process_time


class mission:
    """
    surveying mission

    Survey lines are read in as a pandas dataframe
    then converted to a list of dictionaries

    Filename is formatted as
    wp1_x,wp1_y,wp2_x,wp2_y,speed,offline_std,speed_std,runup,runout

    """

    def __init__(self, survey_line_filename):
        self.survey_line_filename = survey_line_filename

        self.survey_lines = self.read_survey_lines(survey_line_filename)
        self.num_lines = len(self.survey_lines)

    def read_survey_lines(self, survey_line_filename):
        survey_line_data = pd.read_csv(survey_line_filename)
        survey_lines = []

        for i in range(len(survey_line_data["wp1_x"])):
            dist_x = survey_line_data["wp2_x"][i] - survey_line_data["wp1_x"][i]
            dist_y = survey_line_data["wp2_y"][i] - survey_line_data["wp1_y"][i]

            line_dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
            line_az = np.arctan2(dist_y, dist_x)

            line_dictionary = {
                "p1_x": survey_line_data["wp1_x"][i],
                "p1_y": survey_line_data["wp1_y"][i],
                "p2_x": survey_line_data["wp2_x"][i],
                "p2_y": survey_line_data["wp2_y"][i],
                "goal_speed": survey_line_data["speed"][i],
                "offline_std": survey_line_data["offline_std"][i],
                "speed_std": survey_line_data["speed_std"][i],
                "runup": survey_line_data["runup"][i],
                "runout": survey_line_data["runout"][i],
                "distance": line_dist,
                "az": line_az,
            }
            survey_lines.append(line_dictionary)
        return survey_lines

    def plot_survey_lines(self, ax):
        for line in self.survey_lines:
            x = [line["p1_x"], line["p2_x"]]
            y = [line["p1_y"], line["p2_y"]]
            ax.plot(x, y, ".-", color="k")


class fitness:
    """
    'gates' are placed along the line, to compute the fitness
    gate_length is the length in the downline direction
    a fitness is computed for each gate, with a max score of 100

    offline_importance is how much the offline distance weighs vs speed
    (e.g. offline_importance = 0.9 means max speed score can only be 10)

    offline_std and speed_std are used to fit gaussian and give fitness to points
    values > 2 std off line given fitness = 0, gate not counted as fone through
    no gates can be skipped

    start/ end of lines may not have gates, based on runup and runout distances

    fitness function takes current_gate as an input, and starts evaluating that gate
    if data is in that gate, check the next gate
    * if data is in both the first gate, and next gate, only count to first gate
    if data is in that gate, add 1 to current_gate
    populates a np array of fitness values <num_gates,1>

    mission fitness function takes a FITNESS class, returns a FITNESS class
    - num points in gate
    - mean, min, max, std fitness
    - mean, min, max, std velocity
    - mean, min, max, std cross-shore
    - self.current_gate
    - self.current_fitness

    - fitness (mission, boat)
    - self.current_gate
    - self.last_boat_history_ind
    - self.current_fitness
    - self.gate_fitness = np.array([])
    - self.gate_fitness_offline = np.array([])
    - self.gate_fitness_velocity = np.array([])

    """

    def __init__(self, mission, gate_length=1, offline_importance=0.5):
        self.mission = mission

        self.gate_length = gate_length
        self.offline_importance = offline_importance

        self.all_gate = self.create_gates()  # list of dicts
        self.all_gate_fitness = self.create_gate_fitness()  # list of dicts

        self.current_survey_line = 0
        self.current_gate_num = 0
        self.current_fitness = 0
        self.last_boat_history_ind = 0
        self.mission_complete = True

    def create_gate_fitness(self):
        """ Preallocates a list of dicts for each gate """
        # num points in gate
        # mean, min, max fitness
        # mean, min, max downline velocity
        # mean, min, max off-line
        all_gate_fitness = []
        for i in range(len(self.all_gate)):
            gate_data = {
                "npts": 0,
                "mean_fitness": 0,
                "min_fitness": 0,
                "max_fitness": 0,
                "mean_velocity": 0,
                "min_velocity": 0,
                "max_velocity": 0,
                "mean_offline": 0,
                "min_offline": 0,
                "max_offline": 0,
                "fitness": 0,
            }
            all_gate_fitness.append(gate_data)
        return all_gate_fitness

    def create_gates(self):
        """ Create List of Gates with dictionary of elements for each"""
        GATE_STD_MAX_MULT = 2
        all_gate = []
        for line_num, line in enumerate(self.mission.survey_lines):
            p1 = [line["p1_x"], line["p1_y"]]
            p2 = [line["p2_x"], line["p2_y"]]
            line_length = line["distance"]
            downline_gate_pos = np.arange(
                line["runup"], line_length - line["runout"], self.gate_length
            )
            for downline_pos in downline_gate_pos:
                offline_std = line["offline_std"]
                max_offline = offline_std * GATE_STD_MAX_MULT
                speed_std = line["speed_std"]
                # compute polygon corners for the gate zone
                poly_corners = [self.gate_length, max_offline] * np.array(
                    [[0, 1, 1, 0, 0], [1, 1, -1, -1, 1]]
                ).T + [downline_pos, 0]
                world_poly_corners = np.array(
                    self.downline_offline_to_xy(poly_corners, p1, p2)
                ).T
                # compute start of the gate in world coords
                gate_start_pos = np.array([downline_pos, 0]).reshape(1, 2)
                world_xy_start = np.array(
                    self.downline_offline_to_xy(gate_start_pos, p1, p2)
                ).T

                # populate gate
                gate_vals = {
                    "line_az": line["az"],  # azimuth of the survey line teh gate is in
                    "line_num": line_num,  # index for the survey line the gate is in
                    "goal_speed": line["goal_speed"],  # desired speed
                    "downline_pos": downline_pos,  # start of gate in downline coordinates
                    "downline_length": self.gate_length,  # length of gate in downline
                    "offline_std": offline_std,  # std of gaussian to calcoffline fitness
                    "max_offline": max_offline,  # max offline distance (width/2)
                    "speed_error_std": speed_std,  # std of gaussian to calc speed fitness
                    "gate_xy_start": world_xy_start,  # world xy of the start of the gate
                    "gate_xy_polygon": world_poly_corners,  # world xy corners of the gate
                }
                all_gate.append(gate_vals)
        return all_gate

    def update_fitness(self, pos_xy, vel_xy):
        """ Update fitness data """
        # Get only new position and velocity data
        last_ind = self.last_boat_history_ind
        new_pos_xy = pos_xy[last_ind:, :]
        new_vel_xy = vel_xy[last_ind:, :]
        self.last_boat_history_ind = pos_xy.shape[0]

        # While still gates to check
        first_pass = True
        is_more_data = True
        while self.current_gate_num < len(self.all_gate) and is_more_data:
            current_gate = self.all_gate[self.current_gate_num]
            self.current_survey_line = current_gate["line_num"]
            # if first time in the loop, or it's a different survey line, recalc values
            if first_pass or self.current_survey_line != current_gate["line_num"]:
                (
                    downline_pos,
                    offline_pos,
                    downline_vel,
                    offline_vel,
                ) = self.get_pos_vel_gate(
                    new_pos_xy, new_vel_xy, current_gate["line_num"]
                )
            # indicies of data within gate
            d_downline = downline_pos - current_gate["downline_pos"]
            in_gate_downline = np.bitwise_and(
                d_downline >= 0, d_downline < current_gate["downline_length"]
            )
            ind_gate_offline = np.abs(offline_pos) < current_gate["max_offline"]
            ind_in_gate = np.bitwise_and(in_gate_downline, ind_gate_offline)

            # update fitness for all in the gate
            if any(ind_in_gate):
                self.update_fitness_gate(
                    offline_pos[ind_in_gate],
                    downline_vel[ind_in_gate] - current_gate["goal_speed"],
                )
                # remove these points from new data
                ind_other_points = np.bitwise_not(ind_in_gate)
                new_pos_xy = new_pos_xy[ind_other_points, :]
                new_vel_xy = new_vel_xy[ind_other_points, :]
                is_more_data = new_pos_xy.shape[0] > 0
                if is_more_data:  # go to next gate
                    self.current_gate_num += 1
            elif (
                first_pass and self.all_gate_fitness[self.current_gate_num]["npts"] > 0
            ):
                # if the last update ended without going to a new gate
                # but new data started in new gate
                self.current_gate_num += 1
            else:  # got to a gate with no data
                return
        self.mission_complete = True
        return

    def update_fitness_gate(self, offline, velocity):
        """ Update the fitness data in the current gate """
        # number of values in the current sample
        num_new_vals = offline.size
        num_old_vals = self.all_gate_fitness[self.current_gate_num]["npts"]

        # Calculate fitness for every sample
        gate = self.all_gate[self.current_gate_num]
        offline_fitness = self.gaussfun(offline, 0, gate["offline_std"])
        velocity_fitness = self.gaussfun(velocity, 0, gate["speed_error_std"])

        total_fitness = offline_fitness * self.offline_importance + velocity_fitness * (
            1 - self.offline_importance
        )
        # if it's all new data, just populate it directly
        if num_old_vals == 0:
            self.all_gate_fitness[self.current_gate_num] = {
                "npts": num_new_vals,
                "mean_fitness": np.mean(total_fitness),
                "min_fitness": np.min(total_fitness),
                "max_fitness": np.max(total_fitness),
                "mean_velocity": np.mean(velocity_fitness),
                "min_velocity": np.min(velocity_fitness),
                "max_velocity": np.max(velocity_fitness),
                "mean_offline": np.mean(offline_fitness),
                "min_offline": np.min(offline_fitness),
                "max_offline": np.max(offline_fitness),
                "fitness": np.mean(total_fitness),  # change this to max?
            }
            self.current_fitness += self.all_gate_fitness[self.current_gate_num][
                "fitness"
            ]
        else:
            # compute min, max, mean of all
            old_data = self.all_gate_fitness[self.current_gate_num]
            new_npts = num_new_vals + num_old_vals

            new_max_fitness = np.max([np.max(total_fitness), old_data["max_fitness"]])
            new_max_offline = np.max([np.max(offline_fitness), old_data["max_offline"]])
            new_max_vel = np.max([np.max(velocity_fitness), old_data["max_velocity"]])

            new_min_fitness = np.min([np.min(total_fitness), old_data["min_fitness"]])
            new_min_offline = np.min([np.min(offline_fitness), old_data["min_offline"]])
            new_min_vel = np.min([np.min(velocity_fitness), old_data["min_velocity"]])

            new_mean_fitness = self.running_mean(
                total_fitness, old_data["mean_fitness"], num_old_vals
            )
            new_mean_offline = self.running_mean(
                offline_fitness, old_data["mean_offline"], num_old_vals
            )
            new_mean_vel = self.running_mean(
                velocity_fitness, old_data["mean_velocity"], num_old_vals
            )
            fitness_change = new_mean_fitness - old_data["fitness"]

            self.all_gate_fitness[self.current_gate_num] = {
                "npts": new_npts,
                "mean_fitness": new_mean_fitness,
                "min_fitness": new_min_fitness,
                "max_fitness": new_max_fitness,
                "mean_velocity": new_mean_vel,
                "min_velocity": new_min_vel,
                "max_velocity": new_max_vel,
                "mean_offline": new_mean_offline,
                "min_offline": new_min_offline,
                "max_offline": new_max_offline,
                "fitness": new_mean_fitness,  # change this to max?
            }
            self.current_fitness += fitness_change

    @staticmethod
    def running_mean(new_values, old_mean, num_old):
        """ Return the new running average of values """
        new_mean = np.mean(new_values)
        num_new = new_values.size
        total_mean = (num_new * new_mean + num_old * old_mean) / (num_new + num_old)
        return total_mean

    def get_pos_vel_gate(self, pos_xy, vel_xy, line_num):
        line = self.mission.survey_lines[line_num]
        downline_pos, offline_pos = self.xy_to_downline_offline(
            pos_xy, [line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]]
        )
        downline_vel, offline_vel = self.xy_to_downline_offline(
            vel_xy, [line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]], True
        )

        return (downline_pos, offline_pos, downline_vel, offline_vel)

    def plot_gates(self, ax):
        for i, gate in enumerate(self.all_gate):
            if i == self.current_gate_num:
                color = "g"
            elif i < self.current_gate_num:
                color = "r"
            else:
                color = (0.1, 0.1, 0.1)

            ax.add_patch(
                Polygon(
                    gate["gate_xy_polygon"],
                    fc=color,
                    edgecolor="k",
                    zorder=5,
                    alpha=0.5,
                )
            )

    @staticmethod
    def xy_to_downline_offline(xy, p1, p2, do_vel=False):
        """ convert from world xy [Nx2] to downline-offline coordinates """
        az = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        if do_vel:
            downline = (xy[:, 0]) * np.cos(az) + (xy[:, 1]) * np.sin(az)
            offline = (xy[:, 0]) * -np.sin(az) + (xy[:, 1]) * np.cos(az)
        else:
            downline = (xy[:, 0] - p1[0]) * np.cos(az) + (xy[:, 1] - p1[1]) * np.sin(az)
            offline = (xy[:, 0] - p1[0]) * -np.sin(az) + (xy[:, 1] - p1[1]) * np.cos(az)

        return (downline, offline)

    @staticmethod
    def downline_offline_to_xy(xyline, p1, p2, do_vel=False):
        """ convert from xyline [Nx2] to world x,y coordinates """
        az = -np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        if do_vel:
            y = (xyline[:, 0]) * np.cos(az) + (xyline[:, 1]) * np.sin(az)
            x = (xyline[:, 0]) * -np.sin(az) + (xyline[:, 1]) * np.cos(az)
        else:
            y = (xyline[:, 0]) * np.cos(az) + (xyline[:, 1]) * np.sin(az) + p1[0]
            x = (xyline[:, 0]) * -np.sin(az) + (xyline[:, 1]) * np.cos(az) + p1[1]

        return [y, x]

    @staticmethod
    def gaussfun(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# @profile
def main():
    survey_line_filename = (
        "C:/Users/Richie/Documents/GitHub/asvap/data/missions/increasingangle.txt"
    )
    my_mission = mission(survey_line_filename)
    my_fitness = fitness(my_mission, gate_length=1, offline_importance=0.5)

    import boat

    myboat = boat.boat(pos=[0, 0, 0])

    for t in range(100):
        if t == 0:
            myboat.throttle = [55, 58]
        elif t == 4:
            myboat.throttle = [40, 30]
        elif t == 8:
            myboat.throttle = [20, 70]
        elif t == 9:
            myboat.throttle = [55, 30]
        elif t == 10:
            myboat.throttle = [30, 31]
        elif t == 20:
            myboat.throttle = [33, 31]
        elif t == 35:
            myboat.throttle = [40, 41]
        elif t == 65:
            myboat.throttle = [40, 50]
        elif t == 67:
            myboat.throttle = [80, 80]
        elif t == 80:
            myboat.throttle = [80, 40]
        elif t == 82:
            myboat.throttle = [80, 80]
        myboat.update_position(1, 10)

        pos_xy = myboat.history[:, [1, 2]]
        vel_xy = myboat.history[:, [4, 5]]

        my_fitness.update_fitness(pos_xy, vel_xy)

    fig, ax = plt.subplots()
    my_mission.plot_survey_lines(ax)
    myboat.plot_history_line(ax)
    my_fitness.plot_gates(ax)
    ax.set_aspect("equal", "box")
    ax.set_title(f"Fitness = {my_fitness.current_fitness:.2f}")
    plt.show()


if __name__ == "__main__":
    # add @profile above the function you want to profile
    # Then call this script from the command line
    # ".\env\Scripts\kernprof.exe" -l -v ".\apasv\simulator\mission.py" > .\profiler.txt
    t1 = process_time()
    main()
    t2 = process_time()
    print(t2 - t1)
