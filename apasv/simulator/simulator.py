import numpy as np
import pygame


class simulator:
    def __init__(self, boat, environment, visual, fitness, autopilot=None):
        self.boat = boat
        self.environment = environment
        self.visual = visual
        self.fitness = fitness
        self.autopilot = autopilot
        self.keys_pressed = None
        if visual is not None and autopilot is not None:
            self.autopilot.do_partials = True

    def reset(self):
        self.boat.reset()
        self.fitness.reset()
        self.autopilot.survey_lines = self.fitness.mission.survey_lines
        self.autopilot.mission_complete = False
        self.autopilot.current_line = 0

    def get_fitness(self):
        """ Get the fitness of the boat on the mission """
        is_valid_run = self.fitness.is_valid_run(self.boat.history_of_updates)
        if is_valid_run:
            return self.fitness.current_fitness
        else:
            return 0

    def get_mission_waypoints(self):
        """ return the mission """
        return self.fitness.mission.survey_lines

    def get_boat_data(self):
        """ Return Data from the Boat that the autopilot could use """
        # use a dictionary so its more intuitive in autopilot code
        # pos x,y,az
        # vel x,y,az
        boat_data = {
            "x": self.boat.pos["x"],
            "y": self.boat.pos["y"],
            "az": self.boat.pos["az"],
            "vx": self.boat.vel_world["x"],
            "vy": self.boat.vel_world["y"],
            "vaz": self.boat.vel_world["az"],
        }
        return boat_data

    def set_boat_control(self, throttle):
        """ Set the boat throttle """
        self.boat.throttle = throttle

    def update_boat(self, t, n):
        """ update the boat position for t seconds using n steps """
        self.boat.update_position(t, n, self.environment.get_currents)
        t = self.boat.history[:, 0]
        pos_xy = self.boat.history[:, [1, 2]]
        vel_xy = self.boat.history[:, [4, 5]]
        self.fitness.update_fitness(t, pos_xy, vel_xy)

    def update_just_key_input(self):
        self.keys_pressed = self.visual.update()

    def update_visual(self, pause_time=0.1):
        """ Update the pygame visual """
        cam_offset = 200 / 1000 * self.visual.hfov / 2
        self.visual.cam_pos = np.hstack(
            (self.boat.pos["x"] - cam_offset, self.boat.pos["y"])
        )
        self.visual.draw_background()
        self.draw_gate()
        self.visual.draw_grid()
        self.draw_mission()
        self.draw_boat_path()
        self.draw_boat()
        self.draw_currents()
        # self.visual.draw_boat_throttle(self.boat.throttle)
        self.draw_stats()
        self.keys_pressed = self.visual.update()
        pygame.time.delay(int(pause_time * 1000))

    def draw_currents(self):
        current_x, current_y = self.environment.get_currents(
            (self.boat.pos["x"], self.boat.pos["y"])
        )
        boat_x = self.boat.pos["x"]
        boat_y = self.boat.pos["y"]

        line_to_draw = np.array(
            [[boat_x, boat_y], [boat_x + current_x * 10, boat_y + current_y * 10]]
        )

        self.visual.draw_current(line_to_draw)

    def draw_stats(self):
        label_string = []
        data_string = []
        data_val = []
        data_val_range = []
        data_center = []
        data_color_range = []
        data_colors = []

        BOAT_COLOR = [[255, 0, 0], [0, 255, 0]]
        AUTO_COLOR = [[155, 0, 50], [50, 200, 0]]
        PARTIAL_COLOR = [[200, 180, 150], [200, 200, 240]]
        CURRENT_COLORS = [[236, 102, 102], [137, 169, 238]]
        # Throttle Left
        label_string.append("Throttle L")
        data_string.append(f"{self.boat.throttle[0]:+4.0f}")
        data_val.append(self.boat.throttle[0])
        data_val_range.append([-100, 100])
        data_center.append(0.5)
        data_color_range.append([[0, 0.5], [0.5, 1]])
        data_colors.append(BOAT_COLOR)

        # Throttle Right
        label_string.append("Throttle R")
        data_string.append(f"{self.boat.throttle[1]:+4.0f}")
        data_val.append(self.boat.throttle[1])
        data_val_range.append([-100, 100])
        data_center.append(0.5)
        data_color_range.append([[0, 0.5], [0.5, 1]])
        data_colors.append(BOAT_COLOR)

        # Downline Velocity
        if self.autopilot is not None:
            line = self.fitness.mission.survey_lines[self.autopilot.current_line]
        else:
            line = self.fitness.mission.survey_lines[self.fitness.current_survey_line]

        vel_xy = np.hstack(
            (self.boat.vel_world["x"], self.boat.vel_world["y"])
        ).reshape(1, 2)
        downline_vel, offline_vel = self.fitness.xy_to_downline_offline(
            vel_xy, [line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]], True
        )

        label_string.append("Downline Velocity")
        data_string.append(f"{downline_vel[0]:+5.2f}")
        data_val.append(None)
        data_val_range.append(None)
        data_center.append(None)
        data_color_range.append(None)
        data_colors.append(None)

        # Velocity Error
        delta_vel_downline = downline_vel - line["goal_speed"]
        label_string.append("Velocity Error")
        data_string.append(f"{delta_vel_downline[0]:+4.2f}")
        data_val.append(delta_vel_downline[0])
        data_val_range.append([-0.5, 0.5])
        data_center.append(0.5)
        data_color_range.append([[0, 0.5], [0.5, 1]])
        data_colors.append(BOAT_COLOR)

        # pos_offline
        pos_xy = np.hstack((self.boat.pos["x"], self.boat.pos["y"])).reshape(1, 2)
        downline_pos, offline_pos = self.fitness.xy_to_downline_offline(
            pos_xy, [line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]]
        )

        label_string.append("Offline")
        data_string.append(f"{offline_pos[0]:+5.2f}")
        data_val.append(-offline_pos)
        data_val_range.append([-2, 2])
        data_center.append(0.5)
        data_color_range.append([[0, 0.5], [0.5, 1]])
        data_colors.append(BOAT_COLOR)

        # fitness
        if self.fitness.current_gate_num == 0:
            avg_per_gate = 0
        else:
            avg_per_gate = self.get_fitness() / (self.fitness.current_gate_num)

        label_string.append("Fitness")
        data_string.append(f"{self.get_fitness():+5.2f} ({avg_per_gate:3.2f})")
        data_val.append(None)
        data_val_range.append(None)
        data_center.append(None)
        data_color_range.append(None)
        data_colors.append(None)

        # Time
        label_string.append("Time")
        data_string.append(f"{self.boat.time:.1f}")
        data_val.append(None)
        data_val_range.append(None)
        data_center.append(None)
        data_color_range.append(None)
        data_colors.append(None)

        # FPS
        label_string.append("FPS")
        data_string.append(f"{self.visual.fps:.2f}")
        data_val.append(None)
        data_val_range.append(None)
        data_center.append(None)
        data_color_range.append(None)
        data_colors.append(None)

        # CURRENT X
        current_x, current_y = self.environment.get_currents(
            (self.boat.pos["x"], self.boat.pos["y"])
        )
        label_string.append("Current X")
        data_string.append(f"{current_x:+.2f}")
        data_val.append(current_x)
        data_val_range.append([-0.5, 0.5])
        data_center.append(0.5)
        data_color_range.append([[0, 0.5], [0.5, 1]])
        data_colors.append(CURRENT_COLORS)

        # CURRENT Y
        label_string.append("Current Y")
        data_string.append(f"{current_y:+.2f}")
        data_val.append(current_y)
        data_val_range.append([-0.5, 0.5])
        data_center.append(0.5)
        data_color_range.append([[0, 0.5], [0.5, 1]])
        data_colors.append(CURRENT_COLORS)

        # draw all labels
        X_POS = 0
        Y_POS = 0
        LABEL_WIDTH = 150
        DATA_WIDTH = 150
        HEIGHT = 30

        current_y_pos = Y_POS
        self.visual.add_text_rect_data(
            pos=[X_POS, current_y_pos, LABEL_WIDTH + DATA_WIDTH, HEIGHT],
            text_str="BOAT DATA",
        )
        current_y_pos += HEIGHT
        for i in range(len(label_string)):

            self.visual.add_text_rect_data(
                pos=[X_POS, current_y_pos, LABEL_WIDTH, HEIGHT],
                text_str=label_string[i],
            )
            self.visual.add_text_rect_data(
                pos=[X_POS + LABEL_WIDTH, current_y_pos, DATA_WIDTH, HEIGHT],
                text_str=data_string[i],
                data=data_val[i],
                data_range=data_val_range[i],
                bar_relative_start_pos=data_center[i],
                data_color_range=data_color_range[i],
                data_colors=data_colors[i],
            )
            current_y_pos += HEIGHT

        if self.autopilot is not None:
            # add titlebar
            self.visual.add_text_rect_data(
                pos=[X_POS, current_y_pos, LABEL_WIDTH + DATA_WIDTH, HEIGHT],
                text_str="AUTOPILOT DATA",
            )
            current_y_pos += HEIGHT
            # look through each label
            for (label, data, partials) in zip(
                self.autopilot.debug_autopilot_labels,
                self.autopilot.debug_autopilot_label_data,
                self.autopilot.debug_autopilot_partials,
            ):
                # add partials
                if partials is not None:
                    num_partials = len(partials)
                    for i, partial in enumerate(partials):
                        self.visual.add_text_rect_data(
                            pos=[
                                X_POS,
                                current_y_pos + i * HEIGHT * 1 / num_partials,
                                LABEL_WIDTH,
                                HEIGHT * 1 / num_partials,
                            ],
                            text_str=None,
                            fontsize=12,
                            text_color=(150, 150, 150),
                            data=partial,
                            data_range=[-1, 1],
                            bar_relative_start_pos=0.5,
                            outline_color=None,
                            data_colors=PARTIAL_COLOR,
                        )
                        bg_color = None
                else:
                    bg_color = (255, 255, 255)

                # add labels
                self.visual.add_text_rect_data(
                    pos=[X_POS, current_y_pos, LABEL_WIDTH, HEIGHT],
                    text_str=label,
                    bg_color=bg_color,
                )
                # add numbers
                self.visual.add_text_rect_data(
                    pos=[X_POS + LABEL_WIDTH, current_y_pos, DATA_WIDTH, HEIGHT],
                    text_str=f"{data[0]:+.2f}",
                    data=data,
                    data_range=[-1, 1],
                    bar_relative_start_pos=0.5,
                    data_colors=AUTO_COLOR,
                )
                current_y_pos += HEIGHT

    def draw_boat(self):
        boat_poly = self.boat.get_boat_polygon(
            self.boat.pos, scale=self.visual.boat_scale
        )
        self.visual.draw_boat(boat_poly, self.boat.color)

    def draw_mission(self):
        for line_num, line in enumerate(self.fitness.mission.survey_lines):
            highlight_line = False
            if self.autopilot is not None:
                if line_num == self.autopilot.current_line:
                    highlight_line = True

            line_points = np.array(
                [[line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]]]
            )
            self.visual.draw_mission(line_points, highlight_line)

    def draw_gate(self):
        NOLDGATE = 5
        Rold = np.linspace(150, self.visual.bg_color[0], NOLDGATE)
        Gold = np.linspace(150, self.visual.bg_color[1], NOLDGATE)
        Bold = np.linspace(150, self.visual.bg_color[2], NOLDGATE)

        NFUTUREGATE = 5
        Rfuture = np.linspace(150, self.visual.bg_color[0], NFUTUREGATE)
        Gfuture = np.linspace(180, self.visual.bg_color[1], NFUTUREGATE)
        Bfuture = np.linspace(215, self.visual.bg_color[2], NFUTUREGATE)

        for i, gate in enumerate(self.fitness.all_gate):
            d_gate = i - self.fitness.current_gate_num
            if d_gate == 0:
                color = [198, 215, 150]
            elif d_gate < 0 and d_gate >= -NOLDGATE:  # Old
                color = [
                    Rold[-d_gate - 1],
                    Gold[-d_gate - 1],
                    Bold[-d_gate - 1],
                ]
            elif d_gate > 0 and d_gate <= NFUTUREGATE:  # Future
                color = [Rfuture[d_gate - 1], Gfuture[d_gate - 1], Bfuture[d_gate - 1]]
            else:
                continue

            self.visual.draw_gate(gate["gate_xy_polygon"].T, color)

    def draw_boat_path(self):
        boat_path = self.boat.history[:, [1, 2]]
        self.visual.draw_boat_path(boat_path)


if __name__ == "__main__":
    import boat
    import mission
    import environment
    import display

    my_visual = display.display()

    my_boat = boat.boat()
    x = np.array([-5, -5, -3.5, -2, -2, 2, 2, 3.5, 5, 5, 2, 2, -2, -2, -5]) / 10 * 0.7
    y = np.array([-5, 4, 5, 4, 0, 0, 4, 5, 4, -5, -5, 0, 0, -5, -5]) / 10
    my_boat.hullshape = np.array([x, y])
    # my_boat.hullshape = my_boat.hullshape * np.array([0.3, 0.5]).reshape(2, 1)
    mission_name = "./data/missions/increasingangle.txt"
    my_mission = mission.mission(survey_line_filename=mission_name)
    my_fitness = mission.fitness(my_mission, gate_length=0.5, offline_importance=0.5)
    my_fitness.current_gate_num = 0
    my_environment = environment.environment()
    my_environment.get_currents = lambda xy: [0, 0]
    my_simulator = simulator(
        boat=my_boat, fitness=my_fitness, environment=my_environment, visual=my_visual,
    )

    for i in range(50):
        if my_simulator.visual.running:
            # my_boat.throttle = ((2 * np.random.rand(2, 1) - 1) * 100).squeeze()
            if i < 5:
                my_simulator.set_boat_control([81, 80])
            else:
                my_simulator.set_boat_control([61, 62])
            my_simulator.update_boat(1, 10)
            my_simulator.update_visual(0.1)
            if my_simulator.keys_pressed[pygame.K_LEFT]:
                print("LEFT")
    print(my_simulator.get_fitness())
    # fitness_per_gate = []
    # for gate_fitness in my_fitness.all_gate_fitness:
    #     fitness_per_gate.append(gate_fitness["fitness"])
    # plt.plot(fitness_per_gate)
    # plt.show()
