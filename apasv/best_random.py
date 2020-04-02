import numpy as np
import sys
from os import system
import os
from simulator import boat, display, environment, mission, simulator
from autopilot import autopilot
import time
import datetime
from copy import deepcopy
import concurrent.futures

sys.path.insert(0, "../autopilot")

# only print pygame welcome message to screen once on import
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# ------------------- CONSTANTS ----------------------------------
DEBUG = False
DEBUG_DELAY = 0.015
#
NTEST = int(1e9)
MAXSEED = int(1e9)
PARALLEL_CHUNK_SIZE = 5000
NUM_BEST = 10
# random seed
RAND_SEED = 11  # random seed to start building random boats
# boat position
START_POSITION = [0, 0, 0]  # x,y,az
BOAT_TIMESTEP = 1
NUM_SUBSTEPS = 5
MAX_TIME = 1000
MAX_TIME_BETWEEN_SEQUENTIAL_GATE = 10
MAX_TIME_BETWEEN_LINES = 30
CUTOFFS = [
    [5, 0.1],  # time to gate 1
    [50, 20],
    [100, 20],
    [200, 50],
    [300, 75],
]  # [time,fitness]
MISSION_NAME = "./data/missions/increasingangle.txt"
# neural net
NUM_NODES = [30, 30, 30]
ACTIVATION = ["sigmoid", "sigmoid", "sigmoid", "sigmoid"]
WEIGHT_SCALE = 1
BIAS_SCALE = 1
WEIGHT_METHOD = "randn"
BIAS_METHOD = "randn"

# create random seeds
if NTEST == MAXSEED:
    all_rand_seed = range(NTEST)
else:
    rng = np.random.default_rng(RAND_SEED)
    all_rand_seed = rng.choice(MAXSEED, NTEST, replace=False)

all_rand_seed = [
    511690,
    70760,
    269924,
    151699,
    307418,
    165866,
    224968,
    232514,
    100493,
    10277,
]
DEBUG = True
# all_rand_seed[0] = 9461794

# create mission and environment
if DEBUG:
    my_visual = display.display()  # debugging visual
else:
    my_visual = None  # (no visual for performance)


def currents(xy):
    # custom current field
    return [0, 0]


my_mission = mission.mission(survey_line_filename=MISSION_NAME, flip_x=False)
my_environment = environment.environment()  # default no currents

my_environment.get_currents = currents

my_fitness_all = mission.fitness(my_mission, gate_length=1, offline_importance=0.8)
my_boat_all = boat.boat(pos=START_POSITION,)
x = np.array([-5, -5, -3.5, -2, -2, 2, 2, 3.5, 5, 5, 2, 2, -2, -2, -5]) / 10 * 0.7
y = np.array([-5, 4, 5, 4, 0, 0, 4, 5, 4, -5, -5, 0, 0, -5, -5]) / 10
my_boat_all.hullshape = np.array([x, y])


def run_simulation(rand_seed):
    t_start = time.time()
    was_cutoff_checked = [False] * len(CUTOFFS)
    my_boat = deepcopy(my_boat_all)
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
        if my_fitness.current_gate_num > 0 and my_fitness.current_gate_num < len(
            my_fitness.all_gate_fitness
        ):
            t_between_gate = (
                my_boat.time
                - my_fitness.all_gate_fitness[my_fitness.current_gate_num - 1]["time"]
            )
            is_between_lines = (
                my_fitness.all_gate[my_fitness.current_gate_num - 1]["line_num"]
                != my_fitness.all_gate[my_fitness.current_gate_num]["line_num"]
            )
        else:
            t_between_gate = 0
            is_between_lines = True

        good_t_between_gate = (
            t_between_gate < MAX_TIME_BETWEEN_SEQUENTIAL_GATE or is_between_lines
        ) and (t_between_gate < MAX_TIME_BETWEEN_LINES)
        loop_criteria = (
            good_time
            and gates_left
            and missed_cutoff
            and not_ap_complete
            and good_t_between_gate
        )
        if DEBUG:
            my_simulator.update_visual(DEBUG_DELAY)
            if not my_simulator.visual.running:
                print(my_simulator.visual.running)
                break

    time_to_run = time.time() - t_start
    percent_complete = 100 * my_fitness.current_gate_num / len(my_fitness.all_gate)
    t_time = time.time()
    fitness_score = my_simulator.get_fitness()
    mean_offline_fitness = my_simulator.fitness.mean_offline_fitness
    mean_velocity_fitness = my_simulator.fitness.mean_velocity_fitness
    if fitness_score == 0:
        avg_fitness = 0
    else:
        avg_fitness = fitness_score / my_fitness.current_gate_num + 1
        mean_offline_fitness /= my_fitness.current_gate_num + 1
        mean_velocity_fitness /= my_fitness.current_gate_num + 1
    return [
        rand_seed,
        fitness_score,
        percent_complete,
        my_boat.time,
        t_time,
        time_to_run,
        avg_fitness,
        mean_offline_fitness,
        mean_velocity_fitness,
    ]


# TODO Make run script more modular, maybe in a separate py file


def is_cutoffs_good(is_cutoff_checked, t, fitness):
    for time_fitness, is_checked in zip(CUTOFFS, is_cutoff_checked):
        if t > time_fitness[0] and not is_checked:
            if fitness < time_fitness[1]:
                return False
    return True


def update_best_list(sim_data=None, best_list=None):
    """ update or create the list of best performing boats """
    # rand_seed, fitness_score, percent_complete, boat_time, datetime_str, time_to_run
    is_changed = False
    if sim_data is None and best_list is None:
        # preallocate list
        best_list = np.zeros((NUM_BEST, 9))

    elif sim_data is None:
        # no data passed for some reason
        print("No Data Added??")
    else:
        if sim_data[1] > best_list[-1, 1]:
            best_list[-1, :] = sim_data
            sorted_ind = np.argsort(best_list[:, 1])
            best_list = np.flipud(best_list[sorted_ind, :])
            is_changed = True

    return best_list, is_changed


def print_sim_results(sim_data, docr=True, rank=0):
    """ Print results to screen """
    tstr = datetime.datetime.fromtimestamp(sim_data[4]).strftime("%m/%d %I:%M %p")
    print_string = (
        f" {rank+1:10.0f} |"
        + f"{sim_data[1]:8.2f} |"
        + f"{sim_data[6]:9.2f} |"
        + f"{sim_data[7]:9.2f} |"
        + f"{sim_data[8]:9.2f} |"
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
        "       RANK | FITNESS | PER GATE |  OFF FIT |"
        + "  VEL FIT | % COMPLETE | BOAT TIME |"
        + "         SEED |        DATETIME | CPU TIME"
    )
    print(headerstr)
    for rank, sim_data in enumerate(best_list):
        print_sim_results(sim_data, docr=True, rank=rank)


if __name__ == "__main__":
    start_time = time.time()
    print("START")
    best_list, _ = update_best_list()  # preallocate
    if PARALLEL_CHUNK_SIZE > 1 and DEBUG is False:
        current_num = 0
        # wont process the last sub-chunk, but thats fine
        while current_num < len(all_rand_seed):
            current_inds = current_num + np.arange(PARALLEL_CHUNK_SIZE)
            last_ind = current_num + PARALLEL_CHUNK_SIZE
            chunk_seeds = all_rand_seed[current_num:last_ind]
            t_chunk_start = time.time()
            with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:
                chunk_results = executor.map(run_simulation, chunk_seeds)
                for sim_results in chunk_results:
                    best_list, _ = update_best_list(sim_results, best_list)
            print_best_list(best_list)
            t_chunk = time.time() - t_chunk_start
            print(
                f"CHUNK {current_num:,.0f} x {PARALLEL_CHUNK_SIZE:,.0f} in {t_chunk:.2f}s"
            )
            current_num += PARALLEL_CHUNK_SIZE

    else:
        for i, seed in enumerate(all_rand_seed):
            sim_results = run_simulation(seed)
            best_list, is_changed = update_best_list(sim_results, best_list)
            if is_changed:
                print_best_list(best_list)
            else:
                print_sim_results(sim_results, docr=False, rank=i)
            if DEBUG:
                if not my_visual.running:
                    break

    runtime = time.time() - start_time
    print(f"Processed in: {runtime:.3f} s")
