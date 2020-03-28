import numpy as np
import pygame

from simulator import boat, display, environment, mission, simulator

# CONSTANTS
THROTTLE_STEP = 5
BOAT_TIMESTEP = 0.1
VISUAL_DELAY = 0.01

key_list = {
    "left_throttle_up": pygame.K_q,
    "left_throttle_down": pygame.K_a,
    "right_throttle_up": pygame.K_p,
    "right_throttle_down": pygame.K_l,
    "all_throttle_up": pygame.K_UP,
    "all_throttle_down": pygame.K_DOWN,
    "turn_right": pygame.K_RIGHT,
    "turn_left": pygame.K_LEFT,
    "stop": pygame.K_s,
}

is_turn_pressed = False


def update_throttle(keys_pressed, cur_throttle):
    """ Read the keys in and update the throttle accordingly """
    if keys_pressed is not None:

        if keys_pressed[key_list["left_throttle_up"]]:
            cur_throttle[0] += THROTTLE_STEP
        if keys_pressed[key_list["left_throttle_down"]]:
            cur_throttle[0] -= THROTTLE_STEP
        if keys_pressed[key_list["right_throttle_up"]]:
            cur_throttle[1] += THROTTLE_STEP
        if keys_pressed[key_list["right_throttle_down"]]:
            cur_throttle[1] -= THROTTLE_STEP

        if keys_pressed[key_list["all_throttle_up"]]:
            cur_throttle[0] += THROTTLE_STEP
            cur_throttle[1] += THROTTLE_STEP
        if keys_pressed[key_list["all_throttle_down"]]:
            cur_throttle[0] -= THROTTLE_STEP
            cur_throttle[1] -= THROTTLE_STEP
        global is_turn_pressed
        last_turn_pressed = is_turn_pressed
        is_turn_pressed = False
        if keys_pressed[key_list["turn_right"]]:
            cur_throttle[0] += THROTTLE_STEP
            cur_throttle[1] -= THROTTLE_STEP
            is_turn_pressed = True
        if keys_pressed[key_list["turn_left"]]:
            cur_throttle[0] -= THROTTLE_STEP
            cur_throttle[1] += THROTTLE_STEP
            is_turn_pressed = True

        if not is_turn_pressed and last_turn_pressed:
            new_throttle = (cur_throttle[0] + cur_throttle[1]) / 2
            cur_throttle[0] = new_throttle
            cur_throttle[1] = new_throttle

        if keys_pressed[key_list["stop"]]:
            cur_throttle[0] = 0
            cur_throttle[1] = 0

        if cur_throttle[0] > 100:
            cur_throttle[0] = 100
        elif cur_throttle[0] < -100:
            cur_throttle[0] = -100

        if cur_throttle[1] > 100:
            cur_throttle[1] = 100
        elif cur_throttle[1] < -100:
            cur_throttle[1] = -100
    return cur_throttle


my_visual = display.display()

my_boat = boat.boat(friction=[1, 1, 5, 50])
x = np.array([-5, -5, -3.5, -2, -2, 2, 2, 3.5, 5, 5, 2, 2, -2, -2, -5]) / 10 * 0.7
y = np.array([-5, 4, 5, 4, 0, 0, 4, 5, 4, -5, -5, 0, 0, -5, -5]) / 10
my_boat.hullshape = np.array([x, y])
# my_boat.hullshape = my_boat.hullshape * np.array([0.3, 0.5]).reshape(2, 1)
mission_name = (
    "C:/Users/Richie/Documents/GitHub/asvap/data/missions/increasingangle.txt"
)
my_mission = mission.mission(survey_line_filename=mission_name)
my_fitness = mission.fitness(my_mission, gate_length=1, offline_importance=0.5)
# my_fitness.current_gate_num = 10
my_environment = environment.environment()
my_environment.get_currents = lambda xy: [-1, 0]
my_simulator = simulator.simulator(
    boat=my_boat, environment=my_environment, visual=my_visual, fitness=my_fitness,
)


current_throttle = np.array([0, 0])
for x in range(10000):
    if my_simulator.visual.running:
        # my_boat.throttle = ((2 * np.random.rand(2, 1) - 1) * 100).squeeze()
        current_throttle = update_throttle(my_simulator.keys_pressed, current_throttle)
        my_simulator.set_boat_control(current_throttle)

        my_simulator.update_boat(BOAT_TIMESTEP, 10)
        my_simulator.update_visual(VISUAL_DELAY)

print(my_simulator.get_fitness())
