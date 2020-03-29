import numpy as np
import sys
from simulator import boat, display, environment, mission, simulator
from autopilot import autopilot
from time import process_time
import datetime
from copy import deepcopy

sys.path.insert(0, "../autopilot")

DEBUG = True
DEBUG_DELAY = 0.5
# CONSTANTS
NTEST = int(1e7)
RAND_SEED = 14  # random seed to start building random boats
START_POSITION = [0, 0, 10]  # x,y,az
NUM_NODES = [20, 20]
ACTIVATION = ["sigmoid", "sigmoid", "sigmoid"]
WEIGHT_SCALE = 3
BIAS_SCALE = 3
WEIGHT_METHOD = "randn"
BIAS_METHOD = "randn"
BOAT_TIMESTEP = 1
NUM_SUBSTEPS = 10
MAX_TIME = 500
CUTOFFS = [[3, 0.1], [10, 4], [30, 10], [40, 15]]  # [time,fitness]
MISSION_NAME = (
    "C:/Users/Richie/Documents/GitHub/asvap/data/missions/increasingangle.txt"
)

# create random seeds
np.random.seed(RAND_SEED)
all_rand_seed = np.random.choice(NTEST, NTEST, replace=False)
all_rand_seed[0] = all_rand_seed[2370]
# create mission and environment
if DEBUG:
    my_visual = display.display()  # debugging visual
else:
    my_visual = None  # (no visual for performance)

my_mission = mission.mission(survey_line_filename=MISSION_NAME)
my_environment = environment.environment()  # default no currents

my_fitness_all = mission.fitness(my_mission, gate_length=1, offline_importance=0.75)


def run_simulation(rand_seed):
    t_start = process_time()
    was_cutoff_checked = [False] * len(CUTOFFS)
    my_boat = boat.boat(pos=START_POSITION)
    my_fitness = deepcopy(my_fitness_all)
    my_autopilot = autopilot.ap_nn(
        my_mission.survey_lines,
        num_neurons=NUM_NODES,
        output_softmax=False,
        activation_function_names=ACTIVATION,  # "sigmoid", "relu", "tanh"
        rand_seed=rand_seed,
        rand_weights_method=WEIGHT_METHOD,  # "rand","randn","randpm"
        rand_weights_scalar=WEIGHT_SCALE,
        rand_biases_method=BIAS_METHOD,  # "rand","randn","randpm","zero"
        rand_biases_scalar=BIAS_SCALE,
    )
    my_simulator = simulator.simulator(
        boat=my_boat,
        environment=my_environment,
        visual=my_visual,
        fitness=my_fitness,
        autopilot=my_autopilot,
    )

    loop_criteria = True
    while loop_criteria:
        boat_data = my_simulator.get_boat_data()
        new_throttle = my_autopilot.calc_boat_throttle(boat_data)
        my_simulator.set_boat_control(new_throttle)
        my_simulator.update_boat(BOAT_TIMESTEP, NUM_SUBSTEPS)

        good_time = my_boat.time < MAX_TIME
        gates_left = not my_fitness.mission_complete
        missed_cutoff = is_cutoffs_good(
            was_cutoff_checked, my_boat.time, my_simulator.get_fitness()
        )
        not_ap_complete = not my_autopilot.mission_complete
        loop_criteria = good_time and gates_left and missed_cutoff and not_ap_complete
        if DEBUG:
            my_simulator.update_visual(DEBUG_DELAY)
            if not my_simulator.visual.running:
                print(my_simulator.visual.running)
                break

    time_to_run = process_time() - t_start
    percent_complete = 100 * my_fitness.current_gate_num / len(my_fitness.all_gate)
    time_str_format = datetime.datetime.now().strftime("%m/%d %H:%M %p")
    fitness_score = my_simulator.get_fitness()
    return (
        fitness_score,
        percent_complete,
        rand_seed,
        time_str_format,
        time_to_run,
        my_boat.time,
    )


def is_cutoffs_good(is_cutoff_checked, t, fitness):
    for time_fitness, is_checked in zip(CUTOFFS, is_cutoff_checked):
        if t > time_fitness[0] and not is_checked:
            if fitness < time_fitness[1]:
                return False
    return True


if __name__ == "__main__":
    for i, seed in enumerate(all_rand_seed):
        vals = run_simulation(seed)
        if vals[0] > 50:
            print(
                f"{i:5.0f} | Fitness:{vals[0]:5.2f} , Percent:{vals[1]:5.2f}, t:{vals[5]:6.1f}, dt:{vals[4]:5.3f}"
            )
        if DEBUG:
            if not my_visual.running:
                break
