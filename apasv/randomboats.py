import numpy as np
import pygame
import sys
from simulator import boat, display, environment, mission, simulator
from autopilot import autopilot
from time import process_time

sys.path.insert(0, "../autopilot")

# CONSTANTS
THROTTLE_STEP = 5
BOAT_TIMESTEP = 0.1
NUM_SUBSTEPS = 10
VISUAL_DELAY = 0.0001

my_visual = display.display()


mission_name = (
    "C:/Users/Richie/Documents/GitHub/asvap/data/missions/increasingangle.txt"
)
my_mission = mission.mission(survey_line_filename=mission_name)
my_environment = environment.environment()

my_boat = boat.boat(friction=[1, 1, 5, 50], color=[0.3, 0.8, 0.1])
my_fitness = mission.fitness(my_mission, gate_length=1, offline_importance=0.5)
my_simulator = simulator.simulator(
    boat=my_boat, environment=my_environment, visual=my_visual, fitness=my_fitness,
)
my_autopilot = autopilot.ap_nn(
    my_mission.survey_lines,
    num_neurons=[5, 5, 10],
    output_softmax=False,
    activation_function_names=None,  # "sigmoid", "relu", "tanh"
    rand_seed=11,
    rand_weights_method="randpm",  # "rand","randn","randpm"
    rand_weights_scalar=2,
    rand_biases_method="randpm",  # "rand","randn","randpm","zero"
    rand_biases_scalar=2,
)

# BOAT 2
my_boat2 = boat.boat(friction=[1, 1, 5, 50])
x = np.array([-5, -5, -3.5, -2, -2, 2, 2, 3.5, 5, 5, 2, 2, -2, -2, -5]) / 10 * 0.7
y = np.array([-5, 4, 5, 4, 0, 0, 4, 5, 4, -5, -5, 0, 0, -5, -5]) / 10
my_boat2.hullshape = np.array([x, y])

my_fitness2 = mission.fitness(my_mission, gate_length=1, offline_importance=0.5)
my_simulator2 = simulator.simulator(
    boat=my_boat2, environment=my_environment, visual=my_visual, fitness=my_fitness2,
)
my_autopilot2 = autopilot.ap_nn(
    my_mission.survey_lines,
    num_neurons=[5, 10],
    output_softmax=False,
    activation_function_names=None,  # "sigmoid", "relu", "tanh"
    rand_seed=9,
    rand_weights_method="randn",  # "rand","randn","randpm"
    rand_weights_scalar=1,
    rand_biases_method="randn",  # "rand","randn","randpm","zero"
    rand_biases_scalar=1,
)

t1 = process_time()
for x in range(1000):
    if my_simulator.visual.running:
        # my_boat.throttle = ((2 * np.random.rand(2, 1) - 1) * 100).squeeze()
        boat_data = my_simulator.get_boat_data()
        new_throttle = my_autopilot.calc_boat_throttle(boat_data)
        my_simulator.set_boat_control(new_throttle)
        my_simulator.update_boat(BOAT_TIMESTEP, NUM_SUBSTEPS)
        # boat 2
        boat_data2 = my_simulator2.get_boat_data()
        new_throttle2 = my_autopilot2.calc_boat_throttle(boat_data2)
        my_simulator2.set_boat_control(new_throttle2)
        my_simulator2.update_boat(BOAT_TIMESTEP, NUM_SUBSTEPS)
        # draw multiple boats
        my_simulator.update_visual_noblit(VISUAL_DELAY)
        my_simulator2.draw_boat()
        my_simulator.visual.update()
        pygame.time.delay(100)
        my_simulator.update_just_key_input()
print(process_time() - t1)
print(my_simulator2.boat.time)
