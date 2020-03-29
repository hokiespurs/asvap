import numpy as np
import sys
from os import system
from simulator import boat, display, environment, mission, simulator
from autopilot import autopilot
from time import process_time
import time
import datetime
from copy import deepcopy

sys.path.insert(0, "../autopilot")

DEBUG = False
DEBUG_DELAY = 0.015
# ------------------- CONSTANTS ----------------------------------
PARALLEL_CHUNK_SIZE = 100
NUM_BEST = 20
# random seed
RAND_SEED = 14  # random seed to start building random boats
NTEST = int(1e7)
# boat position
START_POSITION = [0, 0, 10]  # x,y,az
BOAT_TIMESTEP = 0.1
NUM_SUBSTEPS = 10
MAX_TIME = 200
CUTOFFS = [[3, 0.1], [10, 4], [30, 10], [40, 15]]  # [time,fitness]
MISSION_NAME = (
    "C:/Users/Richie/Documents/GitHub/asvap/data/missions/increasingangle.txt"
)
# neural net
NUM_NODES = [20, 20]
ACTIVATION = ["sigmoid", "sigmoid", "sigmoid"]
WEIGHT_SCALE = 1
BIAS_SCALE = 1
WEIGHT_METHOD = "randn"
BIAS_METHOD = "randn"

# create random seeds
np.random.seed(RAND_SEED)
all_rand_seed = np.random.choice(NTEST, NTEST, replace=False)
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
    t_time = time.time()
    fitness_score = my_simulator.get_fitness()
    return [
        rand_seed,
        fitness_score,
        percent_complete,
        my_boat.time,
        t_time,
        time_to_run,
    ]


def is_cutoffs_good(is_cutoff_checked, t, fitness):
    for time_fitness, is_checked in zip(CUTOFFS, is_cutoff_checked):
        if t > time_fitness[0] and not is_checked:
            if fitness < time_fitness[1]:
                return False
    return True


def update_best_list(sim_data=None, best_list=None, run_ind=0):
    """ update or create the list of best performing boats """
    # rand_seed, fitness_score, percent_complete, boat_time, datetime_str, time_to_run
    if sim_data is None and best_list is None:
        # preallocate list
        best_list = np.zeros((NUM_BEST, 6))

    elif sim_data is None:
        # no data passed for some reason
        print("No Data Added??")
    else:
        if sim_data[1] > best_list[-1, 1]:
            best_list[-1, :] = sim_data
            sorted_ind = np.argsort(best_list[:, 1])
            best_list = np.flipud(best_list[sorted_ind, :])
            print_best_list(best_list)
        else:
            print_sim_results(sim_data, docr=False, rank=run_ind)
    return best_list


def print_sim_results(sim_data, docr=True, rank=0):
    """ Print results to screen """
    tstr = datetime.datetime.fromtimestamp(sim_data[4]).strftime("%m/%d %H:%M %p")
    print_string = (
        f" {rank+1:4.0f} |"
        + f"{sim_data[1]:8.2f} |"
        + f"{sim_data[2]:11.1f} |"
        + f"{sim_data[3]:10.1f} |"
        + f"{sim_data[0]:13.0f} |"
        + f"{tstr:>16s} |"
        + f"{sim_data[5]:9.3f}"
    )
    if docr:
        print(print_string)
    else:
        print(
            print_string, end="\r",
        )


def print_best_list(best_list):
    """ Print best_list to the screen """
    # clear screen
    system("cls")
    # print header
    headerstr = (
        " RANK | FITNESS | % COMPLETE | BOAT TIME |"
        + "         SEED |        DATETIME | CPU TIME"
    )
    print(headerstr)
    for rank, sim_data in enumerate(best_list):
        print_sim_results(sim_data, docr=True, rank=rank)


if __name__ == "__main__":
    best_list = update_best_list()  # preallocate
    for i, seed in enumerate(all_rand_seed):
        # if PARALLEL_CHUNK_SIZE > 1:
        #     pass
        # else:
        #     pass

        sim_results = run_simulation(seed)
        best_list = update_best_list(sim_results, best_list, i)

        if DEBUG:
            if not my_visual.running:
                break
