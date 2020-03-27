# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import process_time

# TODO mission.py is not well documented, and could use an overhaul


class mission:
    """ surveying mission and fitness function"""

    def __init__(self, waypoint_filename, fitness_spacing=1, offline_importance=0.5):
        # WP1X, WP1Y, WP2X, WP2Y, SPEED, ...
        # MAX_OFFLINE, MAX_SPEED_ERROR
        self.waypoint_filename = waypoint_filename
        self.fitness_spacing = fitness_spacing
        self.survey_lines = self.read_waypoints(waypoint_filename)
        self.num_lines = len(self.survey_lines)
        self.offline_importance = offline_importance
        self.fitness_score = 0
        self.__last_fitness_ind = 0
        self.fitness_table = self.preallocate_fitness(fitness_spacing)
        self.last_line = 0
        self.last_gate = 0

    def read_waypoints(self, waypoint_filename):
        waypoint_data = pd.read_csv(waypoint_filename)
        survey_lines = []

        for wp1_x, wp1_y, wp2_x, wp2_y, speed, offline_std, speed_std in zip(
            waypoint_data["wp1_x"],
            waypoint_data["wp1_y"],
            waypoint_data["wp2_x"],
            waypoint_data["wp2_y"],
            waypoint_data["speed"],
            waypoint_data["offline_std"],
            waypoint_data["speed_std"],
        ):

            line_dist = np.sqrt((wp1_x - wp2_x) ** 2 + (wp1_y - wp2_y) ** 2)
            num_gates = int(np.ceil(line_dist / self.fitness_spacing))
            downline_eval = np.linspace(0, line_dist, num_gates)
            line_dictionary = {
                "P1x": wp1_x,
                "P1y": wp1_y,
                "P2x": wp2_x,
                "P2y": wp2_y,
                "speed": speed,
                "offline_std": offline_std,
                "speed_std": speed_std,
                "downline_eval": downline_eval,
                "num_gates": num_gates,
            }
            survey_lines.append(line_dictionary)
        return survey_lines

    def preallocate_fitness(self, fitness_spacing):
        """ Preallocate the shape of the fitness table """
        fitness_table = []
        line_gate_x = []
        line_gate_y = []
        for line in self.survey_lines:
            num_gates = line["num_gates"]
            fitness_table.append(np.zeros((num_gates, 1)))
            line_gate_x.append(np.linspace(line["P1x"], line["P2x"], num_gates))
            line_gate_y.append(np.linspace(line["P1y"], line["P2x"], num_gates))

        self.line_gate_x = line_gate_x
        self.line_gate_y = line_gate_y

        return fitness_table

    # @profile
    def get_fitness(self, boat, max_line=np.inf, max_skip_gate=10, max_back_gate=3):
        """update and return fitness of the boat on the mission"""
        # only look at new x,y,z points that haven't been evlauated
        last_ind = self.__last_fitness_ind
        pos_xy, vel_xy = (
            boat.history[last_ind:, [1, 2]],
            boat.history[last_ind:, [4, 5]],
        )
        # update last ind
        self.__last_fitness_ind = boat.history.shape[0]

        # loop through each line
        skipped_gates = -np.inf
        for line_num, line in enumerate(self.survey_lines):
            if max_line >= line_num and line_num >= self.last_line:
                # convert to downline-offline
                downline_pos, offline_pos = self.xy_to_downline_offline(
                    pos_xy, [line["P1x"], line["P1y"]], [line["P2x"], line["P2y"]]
                )
                downline_vel, offline_vel = self.xy_to_downline_offline(
                    vel_xy, [line["P1x"], line["P1y"]], [line["P2x"], line["P2y"]], True
                )

                min_downline = np.min(downline_pos)
                max_downline = np.max(downline_pos)
                # evaluate along line
                downline_eval = line["downline_eval"]
                for gate_num, downline_gate_pos in enumerate(downline_eval):
                    # dont search a ton past a point
                    if skipped_gates > max_skip_gate:
                        return self.fitness_score
                        # x = 0
                    is_old_gate = (
                        line_num == self.last_line
                        and gate_num < self.last_gate - max_back_gate
                    )
                    is_no_data = (
                        max_downline < downline_gate_pos - self.fitness_spacing / 2
                        or min_downline > downline_gate_pos + self.fitness_spacing / 2
                    )
                    if not is_old_gate and not is_no_data:
                        downline_error = downline_pos - downline_gate_pos
                        ind = np.logical_and(
                            abs(downline_error) < self.fitness_spacing / 2,
                            (abs(offline_pos) < line["offline_std"] * 3),
                        )
                        if any(ind):

                            self.last_line = line_num
                            self.last_gate = gate_num
                            skipped_gates = 0
                            offline_score = self.gaussfun(
                                offline_pos[ind], 0, line["offline_std"]
                            )
                            velocity_score = self.gaussfun(
                                downline_vel[ind], line["speed"], line["speed_std"]
                            )

                            fitness = (
                                self.offline_importance * offline_score
                                + (1 - self.offline_importance) * velocity_score
                            )
                            max_fitness = np.max(fitness)
                            if max_fitness > self.fitness_table[line_num][gate_num]:
                                self.fitness_score += (
                                    max_fitness - self.fitness_table[line_num][gate_num]
                                )
                                self.fitness_table[line_num][gate_num] = max_fitness
                        else:
                            skipped_gates += 1
                    else:
                        skipped_gates += 1

        return self.fitness_score

    def plot_survey_lines(self, ax):
        for line in self.survey_lines:
            x = [line["P1x"], line["P2x"]]
            y = [line["P1y"], line["P2y"]]
            ax.plot(x, y, ".-", color="k")

    def plot_offline_bounds(self):
        pass

    def plot_fitness(self):
        dist_val = 0
        for fitness_data in self.fitness_table:
            num_fitness = len(fitness_data)
            x = np.linspace(
                dist_val, dist_val + num_fitness * self.fitness_spacing, num_fitness
            )
            y = fitness_data
            plt.plot(x, y, ".-")

    @staticmethod
    def xy_to_downline_offline(xy, p1, p2, do_vel=False):
        az = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        if do_vel:
            downline = (xy[:, 0]) * np.cos(az) + (xy[:, 1]) * np.sin(az)
            offline = (xy[:, 0]) * -np.sin(az) + (xy[:, 1]) * np.cos(az)
        else:
            downline = (xy[:, 0] - p1[0]) * np.cos(az) + (xy[:, 1] - p1[1]) * np.sin(az)
            offline = (xy[:, 0] - p1[0]) * -np.sin(az) + (xy[:, 1] - p1[1]) * np.cos(az)

        return (downline, offline)

    @staticmethod
    def gaussfun(x, mu, sigma):
        return (1 / (sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main():
    waypoint_filename = (
        "C:/Users/Richie/Documents/GitHub/asvap/data/missions/increasingangle.txt"
    )
    my_mission = mission(waypoint_filename, fitness_spacing=0.1, offline_importance=0.5)

    fig, ax = plt.subplots()
    my_mission.plot_survey_lines(ax)
    import boat

    myboat = boat.boat(pos=[0, 0, 0])

    my_mission.get_fitness(myboat)
    dt = 0
    dt_update = 0
    for t in range(200):
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
        t1_start = process_time()
        myboat.update_position(1, 10)
        t1_stop = process_time()
        dt_update += t1_stop - t1_start
        t1_start = process_time()
        fitness = my_mission.get_fitness(myboat)
        t1_stop = process_time()
        dt += t1_stop - t1_start
    print(
        f"Fitness = {fitness[0]:.2f} (Fitness:{dt: .3f}) - (Update:{dt_update: .3f}):"
    )

    myboat.plot_history_line(ax)

    plt.show()


if __name__ == "__main__":
    # ".\env\Scripts\kernprof.exe" -l -v ".\apasv\simulator\mission.py" > .\test2.txt
    main()


# %%
