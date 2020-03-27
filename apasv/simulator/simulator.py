import numpy as np
import pygame

import boat
import mission
import environment


class simulator:
    def __init__(self, boat, mission, environment, visual, show_visual=True):
        self.boat = boat
        self.mission = mission
        self.environment = environment
        self.visual = visual
        self.show_visual = show_visual

    def get_fitness(self):
        """ Get the fitness of the boat on the mission """
        pass

    def get_data(self):
        pass

    def set_boat_data(self):
        pass

    def update_boat(self, t, n):
        self.boat.update_position(t, n, self.environment.get_currents)
        self.mission.get_fitness(self.boat)

    def update_visual(self, pause_time=0.1):
        self.visual.draw_background()
        self.draw_mission()
        self.draw_boat_path()
        self.draw_boat()
        self.visual.draw_boat_throttle(self.boat.throttle)
        self.visual.draw_stats()
        self.visual.update()
        pygame.time.delay(int(pause_time * 1000))

    def draw_boat(self):
        boat_poly = self.boat.get_boat_polygon(
            self.boat.pos, scale=self.visual.boat_scale
        )
        self.visual.draw_boat(boat_poly, self.boat.color)

    def draw_mission(self):
        for line in self.mission.survey_lines:
            line_points = np.array(
                [[line["P1x"], line["P1y"]], [line["P2x"], line["P2y"]]]
            )
            self.visual.draw_mission(line_points)

    def draw_boat_path(self):
        boat_path = self.boat.history[:, [1, 2]]
        self.visual.draw_boat_path(boat_path)


class pygamevisual:
    def __init__(
        self,
        size=[1000, 1000],
        hfov=20,
        cam_pos=[0, 0],
        grid=1,
        grid_color=[0, 0, 50],
        grid_width=1,
        bg_color=[200, 200, 200],
        boat_scale=1,
        mission_line_color=(255, 255, 255),
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
        # self.keys = pygame.key.get_pressed()
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

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

    def draw_mission(self, line_points):
        line_points_game = self.world_to_game(line_points)
        pygame.draw.lines(
            self.win,
            self.mission_line_color,
            False,
            line_points_game,
            self.mission_line_width,
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
        font = pygame.font.Font(None, 30)
        text = font.render(textstr, True, (0, 0, 0))
        text_rect = text.get_rect(center=(pos[0] + pos[2] / 2, pos[1] + pos[3] / 2))
        self.win.blit(text, text_rect)

    def draw_stats(self):
        # TODO pass in stats as argument
        # TODO add self (x,y) position for throttle and stats
        # TODO show fitness score
        # TODO show neural network input variables

        # Time
        t = np.random.rand(1)[0] * 3000
        self.add_text_rect([120, 0, 40, 40], f"t")
        self.add_text_rect([160, 0, 160, 40], f"{t:.1f}s")

        # Velocity magnitude (relative magnitude)
        pos = [160, 40, 160, 40]
        v_mag = np.random.rand(1)[0] * 3
        v_mag_rel = np.random.rand(1)[0] * 3
        self.add_text_rect([120, 40, 40, 40], f"|V|")
        self.add_text_rect(pos, f"{v_mag:.1f} ({v_mag_rel:.1f})")

        # World X (relative X)
        pos = [160, 80, 160, 40]
        v_x = np.random.randn(1)[0] * 3
        v_x_rel = np.random.randn(1)[0] * 3
        self.add_text_rect([120, 80, 40, 40], f"Vx")
        self.add_text_rect(pos, f"{v_x:+.1f} ({v_x_rel:+.1f})")

        # World Y (relative Y)
        pos = [160, 120, 160, 40]
        v_y = np.random.randn(1)[0] * 3
        v_y_rel = np.random.randn(1)[0] * 3
        self.add_text_rect([120, 120, 40, 40], f"Vy")
        self.add_text_rect(pos, f"{v_y:+.1f} ({v_y_rel:+.1f})")

        # World position
        fitness = np.random.rand(1)[0] * 3000
        self.add_text_rect([120, 160, 40, 40], f"F")
        self.add_text_rect([160, 160, 160, 40], f"{fitness:.2f}")
        pass

    def draw_boat_throttle(self, boat_throttle):
        # draw subwindow
        sub_window_width = 120
        sub_window_height = 200
        pygame.draw.rect(
            self.win, (200, 200, 200), (0, 0, sub_window_width, sub_window_height), 0
        )
        pygame.draw.rect(
            self.win, (0, 0, 0), (0, 0, sub_window_width, sub_window_height), 3
        )
        text_height = 40

        # draw throttle boxes
        BARWIDTH = 40
        BARPAD = 10
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
    # x = np.array([-0.5,-0.5,-0.35,-0.2,-0.2,0.2,0.2,0.35,0.5,0.5,0.2,0.2,-0.2,-0.2,-0.5])*0.7
    # y = np.array([-0.5,0.4,0.5,0.4,0,0,0.4,0.5,0.4,-0.5,-0.5,0,0,-0.5,-0.5])
    # my_boat.hullshape=np.array([x,y])
    my_boat.hullshape = my_boat.hullshape * np.array([0.3, 0.5]).reshape(2, 1)
    mission_name = (
        "C:/Users/Richie/Documents/GitHub/asvap/data/missions/increasingangle.txt"
    )
    my_mission = mission.mission(
        waypoint_filename=mission_name, fitness_spacing=0.1, offline_importance=0.5,
    )

    my_environment = environment.environment()
    my_environment.get_currents = lambda xy: [0.1, 0.01]
    my_simulator = simulator(
        boat=my_boat,
        mission=my_mission,
        environment=my_environment,
        visual=my_visual,
        show_visual=True,
    )

    for x in range(1000):
        if my_simulator.visual.running:
            my_visual.cam_pos = np.hstack((my_boat.pos["x"], my_boat.pos["y"]))
            # my_visual.cam_pos = [0, 0]

            my_boat.throttle = ((2 * np.random.rand(2, 1) - 1) * 100).squeeze()

            my_simulator.update_boat(1, 10)
            my_simulator.update_visual(0.0001)
