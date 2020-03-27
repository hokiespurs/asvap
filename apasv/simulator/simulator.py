import numpy as np
import pygame

import boat
import mission
import environment


class simulator:
    def __init__(self, boat, mission, environment, visual):
        self.boat = boat
        self.mission = mission
        self.environment = environment
        self.visual = visual
        self.keys_pressed = None

    def get_fitness(self):
        """ Get the fitness of the boat on the mission """
        return self.mission.fitness_score

    def get_mission_waypoints(self):
        """ return the mission """
        return self.mission.survey_lines

    def get_boat_data(self):
        """ Return Data from the Boat that the autopilot could use """
        # use a dictionary so its more intuitive in autopilot code
        # pos x,y,az
        # vel x,y,az

        boat_data = {
            "x": self.boat.pos["x"][0],
            "y": self.boat.pos["y"][0],
            "az": self.boat.pos["az"][0],
            "vx": self.boat.vel_world["x"][0],
            "vy": self.boat.vel_world["y"][0],
            "vaz": self.boat.vel_world["az"][0],
        }
        return boat_data

    def set_boat_control(self, throttle):
        """ Set the boat throttle """
        self.boat.throttle = throttle

    def update_boat(self, t, n):
        """ update the boat position for t seconds using n steps """
        self.boat.update_position(t, n, self.environment.get_currents)
        self.mission.get_fitness(self.boat)

    def update_visual(self, pause_time=0.1):
        """ Update the pygame visual """
        cam_offset = 200 / 1000 * self.visual.hfov / 2
        self.visual.cam_pos = np.hstack(
            (self.boat.pos["x"] - cam_offset, self.boat.pos["y"])
        )
        self.visual.draw_background()
        self.draw_mission()
        self.draw_boat_path()
        self.draw_boat()
        self.visual.draw_boat_throttle(self.boat.throttle)
        self.draw_stats()
        self.keys_pressed = self.visual.update()
        pygame.time.delay(int(pause_time * 1000))

    def draw_stats(self):
        label_string = []
        data_string = []

        # Time
        label_string.append("time")
        data_string.append(f"{self.boat.time:.1f}")

        # pos_offline
        line = self.mission.survey_lines[self.mission.last_line]
        pos_xy = np.hstack((self.boat.pos["x"], self.boat.pos["y"])).reshape(1, 2)
        downline_pos, offline_pos = self.mission.xy_to_downline_offline(
            pos_xy, [line["P1x"], line["P1y"]], [line["P2x"], line["P2y"]]
        )
        label_string.append("offline")
        data_string.append(f"{offline_pos[0]:+.2f}")

        # vel_downline (delta)
        vel_xy = np.hstack(
            (self.boat.vel_world["x"], self.boat.vel_world["y"])
        ).reshape(1, 2)
        downline_vel, offline_vel = self.mission.xy_to_downline_offline(
            vel_xy, [line["P1x"], line["P1y"]], [line["P2x"], line["P2y"]], True
        )
        label_string.append("vel_line")
        delta_vel_downline = downline_vel - line["speed"]
        data_string.append(f"{downline_vel[0]:.2f} ({delta_vel_downline[0]:+.2f})")

        # fitness
        label_string.append("Fitness")
        data_string.append(f"{self.mission.fitness_score[0]:.1f}")

        # draw all labels
        self.visual.draw_stats(label_string, data_string)

    def draw_boat(self):
        boat_poly = self.boat.get_boat_polygon(
            self.boat.pos, scale=self.visual.boat_scale
        )
        self.visual.draw_boat(boat_poly, self.boat.color)

    def draw_mission(self):
        for line_num, line in enumerate(self.mission.survey_lines):
            highlight_line = False
            if line_num == self.mission.last_line:
                highlight_line = True
            line_points = np.array(
                [[line["P1x"], line["P1y"]], [line["P2x"], line["P2y"]]]
            )
            self.visual.draw_mission(line_points, highlight_line)

    def draw_boat_path(self):
        boat_path = self.boat.history[:, [1, 2]]
        self.visual.draw_boat_path(boat_path)


class pygamevisual:
    def __init__(
        self,
        size=[1200, 1000],
        hfov=24,
        cam_pos=[0, 0],
        grid=1,
        grid_color=[0, 0, 50],
        grid_width=1,
        bg_color=[200, 200, 200],
        boat_scale=1,
        mission_line_color=(255, 255, 255),
        mission_line_color_highlight=(200, 100, 0),
        mission_line_width=5,
        boat_history_color=(100, 100, 255),
        boat_history_width=4,
    ):
        self.size = np.array(size)
        self.hfov = hfov
        self.scale_factor = self.size[0] / self.hfov
        self.grid = grid
        self.grid_color = grid_color
        self.grid_width = grid_width
        self.bg_color = bg_color
        self.boat_scale = boat_scale
        self.mission_line_color = mission_line_color
        self.mission_line_color_highlight = mission_line_color_highlight
        self.mission_line_width = mission_line_width
        self.boat_history_color = boat_history_color
        self.boat_history_width = boat_history_width
        self.cam_pos = np.array(cam_pos)
        self.running = True
        self.initialize_window()

    def draw_background(self):
        self.win.fill(self.bg_color)
        self.draw_grid()

    def update(self):
        keys_pressed = pygame.key.get_pressed()

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        return keys_pressed

    def initialize_window(self):
        pygame.init()
        self.win = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Simulator")

    def world_to_game(self, xy):
        # Mx2
        return (xy - self.cam_pos) * [
            self.scale_factor,
            -self.scale_factor,
        ] + self.size / 2

    def draw_grid(self):
        """ Draw the grid pattern on the screen"""
        if self.grid is not None:
            # compute the total view of the camera
            x_view = self.size[0] / self.scale_factor
            y_view = self.size[1] / self.scale_factor
            # compute the vector of x values, rounded to nearest integer of grid
            first_x_gridline = (
                np.round((self.cam_pos[0] - x_view / 2) / self.grid) * self.grid
            )
            last_x_gridline = first_x_gridline + x_view + self.grid

            x = np.arange(first_x_gridline, last_x_gridline, self.grid)

            # compute the vector of x values, rounded to nearest integer of grid
            first_y_gridline = (
                np.round((self.cam_pos[1] - y_view / 2) / self.grid) * self.grid
            )
            last_y_gridline = first_y_gridline + y_view + self.grid

            y = np.arange(first_y_gridline, last_y_gridline, self.grid)

            # plot x grid lines
            for ix in x:
                xplot = self.world_to_game(np.array([ix, 0]))[0]
                pygame.draw.line(
                    self.win,
                    self.grid_color,
                    [xplot, 0],
                    [xplot, self.size[1]],
                    self.grid_width,
                )
            # plot y grid lines
            for iy in y:
                yplot = self.world_to_game(np.array([0, iy]))[1]
                pygame.draw.line(
                    self.win,
                    self.grid_color,
                    [0, yplot],
                    [self.size[0], yplot],
                    self.grid_width,
                )

    def draw_boat(self, poly, color=(1, 1, 1)):
        boatcolor = np.array(color) * 255
        poly_game = self.world_to_game(poly.T)
        pygame.draw.polygon(self.win, boatcolor, poly_game)

    def draw_mission(self, line_points, highlight_line):
        line_points_game = self.world_to_game(line_points)
        if highlight_line:
            color = self.mission_line_color_highlight
        else:
            color = self.mission_line_color
        pygame.draw.lines(
            self.win, color, False, line_points_game, self.mission_line_width,
        )

    def draw_boat_path(self, boat_path):
        boat_path_game = self.world_to_game(boat_path)
        pygame.draw.lines(
            self.win,
            self.boat_history_color,
            False,
            boat_path_game,
            self.boat_history_width,
        )

    def add_text_rect(self, pos, textstr):
        pygame.draw.rect(self.win, (200, 200, 200), pos, 0)
        pygame.draw.rect(self.win, (0, 0, 0), pos, 3)
        font = pygame.font.Font(None, 24)
        text = font.render(textstr, True, (0, 0, 0))
        text_rect = text.get_rect(center=(pos[0] + pos[2] / 2, pos[1] + pos[3] / 2))
        self.win.blit(text, text_rect)

    def draw_stats(self, label_str, data_str):
        y_top = 200
        y_height = 30
        for label, data in zip(label_str, data_str):
            self.add_text_rect([0, y_top, 80, y_height], label)
            self.add_text_rect([80, y_top, 120, y_height], data)
            y_top += y_height

    def draw_boat_throttle(self, boat_throttle):
        # draw subwindow
        sub_window_width = 200
        sub_window_height = 200
        pygame.draw.rect(
            self.win, (200, 200, 200), (0, 0, sub_window_width, sub_window_height), 0
        )
        pygame.draw.rect(
            self.win, (0, 0, 0), (0, 0, sub_window_width, sub_window_height), 3
        )
        text_height = 40

        # draw throttle boxes
        BARPAD = 10
        BARWIDTH = sub_window_width / 2 - BARPAD * 2
        throttle_box_range = sub_window_height / 2 - text_height / 2 - BARPAD
        throttle_box_height = np.array(boat_throttle) * throttle_box_range / 100
        bar_x_center = [sub_window_width * 0.25, sub_window_width * 0.75]
        bar_y_center = [sub_window_height / 2] * 2
        for bar_height, x_center, y_start in zip(
            throttle_box_height, bar_x_center, bar_y_center
        ):
            if bar_height > 0:  # Positive Throttle
                bar_color = (0, 255, 0)
                barpos = (
                    x_center - BARWIDTH / 2,
                    y_start - text_height / 2 - bar_height,
                    BARWIDTH,
                    bar_height,
                )
                pygame.draw.rect(self.win, bar_color, barpos, 0)
            elif bar_height < 0:  # Negative Throttle
                bar_color = (255, 0, 0)
                barpos = (
                    x_center - BARWIDTH / 2,
                    y_start + text_height / 2,
                    BARWIDTH,
                    -bar_height,
                )
                pygame.draw.rect(self.win, bar_color, barpos, 0)

        # outline the box
        pygame.draw.rect(
            self.win,
            (0, 0, 0),
            (0, sub_window_height / 2 - text_height / 2, sub_window_width, text_height),
            3,
        )
        pygame.draw.line(
            self.win,
            (0, 0, 0),
            [sub_window_width / 2, 0],
            [sub_window_width / 2, sub_window_height],
            3,
        )
        # add text
        font = pygame.font.Font(None, 30)
        text = font.render(f"{boat_throttle[0]:+3.0f}", True, (0, 0, 0))
        text_rect = text.get_rect(center=(sub_window_width * 0.25, 100))
        self.win.blit(text, text_rect)

        text = font.render(f"{boat_throttle[1]:+3.0f}", True, (0, 0, 0))
        text_rect = text.get_rect(center=(sub_window_width * 0.75, 100))
        self.win.blit(text, text_rect)


if __name__ == "__main__":
    my_visual = pygamevisual()

    my_boat = boat.boat()
    x = np.array([-5, -5, -3.5, -2, -2, 2, 2, 3.5, 5, 5, 2, 2, -2, -2, -5]) / 10 * 0.7
    y = np.array([-5, 4, 5, 4, 0, 0, 4, 5, 4, -5, -5, 0, 0, -5, -5]) / 10
    my_boat.hullshape = np.array([x, y])
    # my_boat.hullshape = my_boat.hullshape * np.array([0.3, 0.5]).reshape(2, 1)
    mission_name = (
        "C:/Users/Richie/Documents/GitHub/asvap/data/missions/increasingangle.txt"
    )
    my_mission = mission.mission(
        waypoint_filename=mission_name, fitness_spacing=0.5, offline_importance=0.5,
    )

    my_environment = environment.environment()
    my_environment.get_currents = lambda xy: [0, 0]
    my_simulator = simulator(
        boat=my_boat, mission=my_mission, environment=my_environment, visual=my_visual,
    )

    for x in range(1000):
        if my_simulator.visual.running:
            # my_boat.throttle = ((2 * np.random.rand(2, 1) - 1) * 100).squeeze()
            my_simulator.set_boat_control([60, 60])

            my_simulator.update_boat(1, 10)
            my_simulator.update_visual(0.01)
            if my_simulator.keys_pressed[pygame.K_LEFT]:
                print("LEFT")
    print(my_simulator.get_fitness())
