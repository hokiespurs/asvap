from dask.distributed import Client
from autopilot import autopilot, genetic
from simulator import mission
from datetime import datetime
import numpy as np
import runautopilots
from copy import deepcopy
import time
import string
import random

START_SEEDS = [
    211478023,
    576029257,
    51788801,
    813176530,
    457325819,
    765444478,
    692363662,
    393353213,
    423721455,
    162562018,
]
NUM_TOP_KEEP = 10
NUM_MUTATE_PER_TOP = 100
MUTATE_PROBABILITY = 0.01
MUTATE_DISTRIBUTION = "randn"
MUTATE_SCALAR = 1

NUM_PER_WORKER = 50
RAND_SEED = 1
SAVE_FOLDER = "./data/batchruns/genetic_test2"

MAX_ITERATIONS = int(1e7)


def id_generator(lastid="0", size=4, chars=string.ascii_uppercase + string.digits):

    time_str = datetime.now().strftime("%H%M")
    return (
        lastid[0] + "-" + "".join(random.choice(chars) for _ in range(size)) + time_str
    )


def mutate_autopilots(
    ap_parent, num_mutations, probability, distribution, scalar, random_generator
):
    """ return a list of mutated autopilots """

    mutated_autopilot_list = []

    for _ in range(num_mutations):
        dna = ap_parent.nn.get_nn_vector()
        mutated_dna = genetic.random_mutation(
            dna, probability, distribution, scalar, random_generator
        )
        mutated_ap = deepcopy(ap_parent)
        mutated_ap.nn.set_nn_vector(mutated_dna)
        mutated_ap.id = id_generator(mutated_ap.id)
        mutated_autopilot_list.append(mutated_ap)

    return mutated_autopilot_list


if __name__ == "__main__":
    random_generator = np.random.default_rng(RAND_SEED)
    MISSION_NAME = "./data/missions/line.txt"
    # autopilot parameters
    autopilot_params = {
        "num_neurons": [30, 30],
        "rand_weights_method": "randn",  # "rand","randn","randpm"
        "rand_weights_scalar": 1,
        "rand_biases_method": "randn",  # "rand","randn","randpm","zero"
        "rand_biases_scalar": 1,
    }
    # class parameters for simulation
    class_params = {
        "boat_params": {},
        "mission_params": {"survey_line_filename": MISSION_NAME, "flip_x": False},
        "environment_params": {"currents_data": "line"},
        "fitness_params": {"gate_length": 1, "offline_importance": 0.8},
        "display_params": {},
        "autopilot_params": autopilot_params,
        "autopilot_type": "genetic",
    }
    # simulation parameters
    simulation_params = {
        "timestep": 1,
        "num_substeps": 5,
        "do_visual": False,
        "visual_timestep": 0.001,
        "cutoff_max_time": 2000,
        "cutoff_time_gates_same_line": 15,
        "cutoff_time_gates_different_line": 30,
        "cutoff_thresh": [[5, 0.1], [100, 0.1]],
    }
    # initiailze top autopilots
    my_mission = mission.mission(**class_params["mission_params"])
    ap_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    top_ap = []

    for i, seed in enumerate(START_SEEDS):
        new_ap = autopilot.ap_nn(
            my_mission.survey_lines, rand_seed=seed, **autopilot_params
        )
        new_ap.id = ap_names[i]
        top_ap.append(new_ap)

    # initiailze best list
    best_list = runautopilots.reset_best_simulations(NUM_TOP_KEEP)

    # Start parallel processing
    client = Client()

    total_runs = 0
    t_start_all = time.time()
    for run_num in range(MAX_ITERATIONS):
        t_start = time.time()
        # make new list of autopilots to test
        all_autopilot_list = top_ap
        for ap in top_ap:
            # mutate each of the top autopilots
            ap_list = mutate_autopilots(
                ap,
                NUM_MUTATE_PER_TOP,
                MUTATE_PROBABILITY,
                MUTATE_DISTRIBUTION,
                MUTATE_SCALAR,
                random_generator,
            )
            all_autopilot_list = all_autopilot_list + ap_list

        # run each of the simulations
        new_runs = runautopilots.run_autopilots_parallel(
            class_params,
            simulation_params,
            all_autopilot_list,
            client,
            num_per_worker=NUM_PER_WORKER,
        )
        # Sort the simulation
        sorted_runs = sorted(enumerate(new_runs), key=lambda x: x[1]["fitness"])
        top_inds = []
        for i in range(1, NUM_TOP_KEEP + 1):
            top_inds.append(sorted_runs[-i][0])

        # get the top N
        top_ap = [all_autopilot_list[i] for i in top_inds]

        # print the top N
        best_list = runautopilots.update_best_simulations(
            new_runs, num_bests=NUM_TOP_KEEP
        )
        runautopilots.print_best_runs(best_list)
        print(f"iteration:{run_num+1}")
        total_runs += len(sorted_runs)
        t_per_boat = (time.time() - t_start) / len(sorted_runs)
        tstr = runautopilots.timer_str(t_start, time.time())
        print(f"{len(sorted_runs):,.0f} in {tstr} [{t_per_boat:.3f}]")
        t_per_boat = (time.time() - t_start_all) / total_runs
        tstr = runautopilots.timer_str(t_start_all, time.time())
        print(f"{total_runs:,.0f} in {tstr} [{t_per_boat:.3f}]")
        # save the top N
        runautopilots.save_autopilot_list(top_ap, SAVE_FOLDER)
        runautopilots.print_best_runs(best_list, SAVE_FOLDER + "/top.txt")

    # http://localhost:8787/status
    # TODO profile for speed, why so much slower than random boats
