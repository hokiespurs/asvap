import numpy as np
import pygame


class simulator:
    def __init__(self, boat, environment, visual, fitness):
        self.boat = boat
        self.environment = environment
        self.visual = visual
        self.fitness = fitness
        self.keys_pressed = None

    def get_fitness(self):
        """ Get the fitness of the boat on the mission """
        return self.fitness.current_fitness

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
        pos_xy = self.boat.history[:, [1, 2]]
        vel_xy = self.boat.history[:, [4, 5]]
        self.fitness.update_fitness(pos_xy, vel_xy)

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
        self.visual.draw_boat_throttle(self.boat.throttle)
        self.draw_stats()
        self.keys_pressed = self.visual.update()
        pygame.time.delay(int(pause_time * 1000))

    def update_visual_noblit(self, pause_time=0.1):
        """ Update the pygame visual """
        self.visual.draw_background()
        self.draw_gate()
        self.visual.draw_grid()
        self.draw_mission()
        self.draw_boat_path()
        self.draw_boat()
        self.visual.draw_boat_throttle(self.boat.throttle)
        self.draw_stats()
        # self.keys_pressed = self.visual.update()
        # pygame.time.delay(int(pause_time * 1000))

    def draw_stats(self):
        label_string = []
        data_string = []

        # Time
        label_string.append("time")
        data_string.append(f"{self.boat.time:.1f}")

        # pos_offline
        line = self.fitness.mission.survey_lines[self.fitness.current_survey_line]
        pos_xy = np.hstack((self.boat.pos["x"], self.boat.pos["y"])).reshape(1, 2)
        downline_pos, offline_pos = self.fitness.xy_to_downline_offline(
            pos_xy, [line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]]
        )
        label_string.append("offline")
        data_string.append(f"{offline_pos[0]:+.2f}")

        # vel_downline (delta)
        vel_xy = np.hstack(
            (self.boat.vel_world["x"], self.boat.vel_world["y"])
        ).reshape(1, 2)
        downline_vel, offline_vel = self.fitness.xy_to_downline_offline(
            vel_xy, [line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]], True
        )
        label_string.append("vel_line")
        delta_vel_downline = downline_vel - line["goal_speed"]
        data_string.append(f"{downline_vel[0]:.2f} ({delta_vel_downline[0]:+.2f})")

        # fitness
        label_string.append("Fitness")
        if self.fitness.current_gate_num == 0:
            avg_per_gate = 0
        else:
            avg_per_gate = self.fitness.current_fitness / (
                self.fitness.current_gate_num
            )
        data_string.append(f"{self.fitness.current_fitness:.1f} ({avg_per_gate:.2f})")

        # fitness
        label_string.append("fps")
        data_string.append(f"{self.visual.fps:.1f}")

        # draw all labels
        self.visual.draw_stats(label_string, data_string)

    def draw_boat(self):
        boat_poly = self.boat.get_boat_polygon(
            self.boat.pos, scale=self.visual.boat_scale
        )
        self.visual.draw_boat(boat_poly, self.boat.color)

    def draw_mission(self):
        for line_num, line in enumerate(self.fitness.mission.survey_lines):
            highlight_line = False
            if line_num == self.fitness.current_survey_line:
                highlight_line = True
            line_points = np.array(
                [[line["p1_x"], line["p1_y"]], [line["p2_x"], line["p2_y"]]]
            )
            self.visual.draw_mission(line_points, highlight_line)

    def draw_gate(self):
        NOLDGATE = 5
        Rold = np.linspace(215, self.visual.bg_color[0], NOLDGATE)
        Gold = np.linspace(151, self.visual.bg_color[1], NOLDGATE)
        Bold = np.linspace(151, self.visual.bg_color[2], NOLDGATE)

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
    mission_name = (
        "C:/Users/Richie/Documents/GitHub/asvap/data/missions/increasingangle.txt"
    )
    my_mission = mission.mission(survey_line_filename=mission_name)
    my_fitness = mission.fitness(my_mission, gate_length=0.5, offline_importance=0.5)
    my_fitness.current_gate_num = 15
    my_environment = environment.environment()
    my_environment.get_currents = lambda xy: [0, 0]
    my_simulator = simulator(
        boat=my_boat, fitness=my_fitness, environment=my_environment, visual=my_visual,
    )

    for x in range(1000):
        if my_simulator.visual.running:
            # my_boat.throttle = ((2 * np.random.rand(2, 1) - 1) * 100).squeeze()
            my_simulator.set_boat_control([60, 60])

            my_simulator.update_boat(1, 10)
            my_simulator.update_visual(0.1)
            if my_simulator.keys_pressed[pygame.K_LEFT]:
                print("LEFT")
    print(my_simulator.get_fitness())
