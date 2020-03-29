import numpy as np
import pygame
from time import process_time


class display:
    """ pygame based display of the data for the simulator"""

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
        boat_history_max_num=3000,
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
        self.boat_history_max_num = boat_history_max_num
        self.cam_pos = np.array(cam_pos)
        self.running = True
        self.initialize_window()

        self.last_update_time = 0
        self.fps = 0

    def draw_background(self):
        self.win.fill(self.bg_color)

    def update(self):
        keys_pressed = pygame.key.get_pressed()
        dt = process_time() - self.last_update_time
        if dt == 0:
            dt = 0.01
        self.fps = (self.fps) * 0.99 + (1 / (dt)) * 0.01
        self.last_update_time = process_time()
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

    def draw_gate(self, poly, color):
        poly_game = self.world_to_game(poly.T)
        pygame.draw.polygon(self.win, color, poly_game)

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
        max_num = self.boat_history_max_num
        if boat_path.shape[0] > max_num:
            boat_path_game = self.world_to_game(boat_path[-max_num:, :])
        else:
            boat_path_game = self.world_to_game(boat_path)
        pygame.draw.lines(
            self.win,
            self.boat_history_color,
            False,
            boat_path_game,
            self.boat_history_width,
        )

    def add_text_rect(self, pos, textstr, doBG=True):
        if doBG:
            pygame.draw.rect(self.win, (200, 200, 200), pos, 0)
        pygame.draw.rect(self.win, (0, 0, 0), pos, 3)
        font = pygame.font.Font(None, 24)
        text = font.render(textstr, True, (0, 0, 0))
        text_rect = text.get_rect(center=(pos[0] + pos[2] / 2, pos[1] + pos[3] / 2))
        self.win.blit(text, text_rect)

    def draw_ap_stats(self, all_labels, all_label_data):
        y_top = 530
        y_height = 30
        bar_length = 60
        bar_center = 140
        self.add_text_rect([0, 500, 200, y_height], "AUTOPILOT")
        for label, data in zip(all_labels, all_label_data):
            self.add_text_rect([0, y_top, 80, y_height], label)
            self.add_text_rect([80, y_top, 120, y_height], f"{data[0]:+4.2f}")
            if data < 0:
                color = [100, 50, 50]
            elif data > 0:
                color = [50, 100, 50]
            pos = [bar_center, y_top, data * bar_length, y_height]
            pygame.draw.rect(self.win, color, pos, 0)
            self.add_text_rect([80, y_top, 120, y_height], f"{data[0]:+4.2f}", False)
            y_top += y_height

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
