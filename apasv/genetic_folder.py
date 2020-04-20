from dask.distributed import Client
from autopilot import genetic

# from simulator import mission
from datetime import datetime
import numpy as np
import runautopilots
from copy import deepcopy
import time
import string
import random
from pathlib import Path

AP_FOLDER = "./data/batchruns/AP_no_currents"
NUM_TOP_KEEP = 10
NUM_MUTATE_PER_TOP = 50
NUM_PER_WORKER = 10
RAND_SEED = 1
MAX_ITERATIONS = int(1e7)
MISSION_NAME = "./data/missions/increasingangle.txt"
MISSION_CURRENTS = None
OFFLINE_IMPORTANCE = 0.2
FITNESS_INCREASE_THRESH = 1
NUM_FITNESS_INCREASE_FAIL_BEFORE_BREAK = 3
MUTATE_PARAMS = [
    [{"probability": 0.01, "distribution": "randn", "mutate_type": "zero"}],
    [{"probability": 0.01, "distribution": "randn", "mutate_type": "sum"}],
    [{"probability": 0.01, "distribution": "randn", "mutate_type": "replace"}],
    [
        {"probability": 0.005, "distribution": "randn", "mutate_type": "replace"},
        {"probability": 0.005, "distribution": "randn", "mutate_type": "sum"},
        {"probability": 0.005, "distribution": "randn", "mutate_type": "zero"},
    ],
]


def id_generator(lastid="0", size=4, chars=string.ascii_uppercase + string.digits):

    time_str = datetime.now().strftime("%H%M")
    return (
        lastid[0] + "-" + "".join(random.choice(chars) for _ in range(size)) + time_str
    )


def mutate_autopilots(ap_parent, num_mutations, random_generator):
    """ return a list of mutated autopilots """
    mutated_autopilot_list = []
    dna = ap_parent.nn.get_nn_vector()
    for _ in range(NUM_MUTATE_PER_TOP):
        for i, params in enumerate(MUTATE_PARAMS):
            ap_dna = dna
            for param in params:
                ap_dna = genetic.random_mutation(ap_dna, random_generator, **param)
            mutated_ap = deepcopy(ap_parent)
            mutated_ap.nn.set_nn_vector(ap_dna)
            # make id count the number of each mutation
            start_ind = 2 + i * 4
            end_ind = 2 + i * 4 + 3
            n = int(mutated_ap.id[start_ind:end_ind])
            mutated_ap.id = (
                mutated_ap.id[0:start_ind] + f"{n+1:03.0f}" + mutated_ap.id[end_ind:]
            )
            mutated_autopilot_list.append(mutated_ap)

    return mutated_autopilot_list


if __name__ == "__main__":
    random_generator = np.random.default_rng(RAND_SEED)
    # class parameters for simulation
    class_params = {
        "boat_params": {},
        "mission_params": {"survey_line_filename": MISSION_NAME, "flip_x": False},
        "environment_params": {"currents_data": MISSION_NAME},
        "fitness_params": {"gate_length": 1, "offline_importance": OFFLINE_IMPORTANCE},
        "display_params": {},
        "autopilot_params": {},
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
    # make save directory within AP_FOLDER
    save_dir = AP_FOLDER + "/genetic"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # read autopilot list from folder
    autopilot_list = runautopilots.load_autopilot_list(AP_FOLDER)
    # Start parallel processing
    client = Client()
    # initiailze top autopilots
    id_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    best_of_best = []
    for ap_num, ap in enumerate(autopilot_list):
        ap.id = id_str[ap_num] + "-000" * len(MUTATE_PARAMS)
        top_ap = [ap]
        # initiailze best list
        best_list = runautopilots.reset_best_simulations(NUM_TOP_KEEP)

        total_runs = 0
        t_start_all = time.time()
        last_fitness = 0
        num_fitness_increase_fail = 0
        for run_num in range(MAX_ITERATIONS):
            t_start = time.time()
            # make new list of autopilots to test
            all_autopilot_list = top_ap
            for ap_parent in top_ap:
                # mutate each of the top autopilots
                ap_list = mutate_autopilots(
                    ap_parent, NUM_MUTATE_PER_TOP, random_generator
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

            # get the  best fitness
            top_fitness = sorted_runs[-1][1]["fitness"]
            # compare the mean best fitness with past best fitness
            # if not getting better by a threshold add n to bad counter
            dfitness = top_fitness - last_fitness
            last_fitness = top_fitness
            if dfitness < FITNESS_INCREASE_THRESH:
                num_fitness_increase_fail += 1
            else:
                num_fitness_increase_fail = 0

            # print the top N
            best_list = runautopilots.update_best_simulations(
                new_runs, num_bests=NUM_TOP_KEEP
            )
            runautopilots.print_best_runs(best_list)
            print(f"iteration:{run_num+1}")
            print(f"Boat {ap_num:.0f} / {len(autopilot_list):.0f}")
            total_runs += len(sorted_runs)
            t_per_boat = (time.time() - t_start) / len(sorted_runs)
            tstr = runautopilots.timer_str(t_start, time.time())
            print(f"{len(sorted_runs):,.0f} in {tstr} [{t_per_boat:.3f}]")
            t_per_boat = (time.time() - t_start_all) / total_runs
            tstr = runautopilots.timer_str(t_start_all, time.time())
            print(f"{total_runs:,.0f} in {tstr} [{t_per_boat:.3f}]")
            print(
                f"Fitness increase: {dfitness:.2f} , "
                + f"({num_fitness_increase_fail:.0f} < "
                + f"{NUM_FITNESS_INCREASE_FAIL_BEFORE_BREAK:.0f})"
            )
            # save the top N
            save_subdir = f"{ap_num:03.0f}"
            runautopilots.save_autopilot_list(top_ap, save_dir + "/" + save_subdir)
            runautopilots.print_best_runs(
                best_list, save_dir + "/" + save_subdir + "/top.txt"
            )

            # if bad counter > thresh, break loop and go to next one
            if num_fitness_increase_fail == NUM_FITNESS_INCREASE_FAIL_BEFORE_BREAK:
                break
        # append best to the best of the best list
        top_ap[0].id += f"-{run_num:.0f}"
        best_of_best.append(top_ap[0])

        # run each of the best simulations
        best_runs = runautopilots.run_autopilots_parallel(
            class_params,
            simulation_params,
            best_of_best,
            client,
            num_per_worker=NUM_PER_WORKER,
        )
        # Sort the best of the best
        sorted_runs = sorted(enumerate(best_runs), key=lambda x: x[1]["fitness"])
        top_inds = []
        for i in range(len(sorted_runs)):
            top_inds.append(sorted_runs[-i][0])

        # print the best of the best
        best_list = runautopilots.update_best_simulations(
            best_runs, num_bests=NUM_TOP_KEEP
        )
        runautopilots.print_best_runs(best_list)
        runautopilots.print_best_runs(best_list, save_dir + "/top.txt")
        # save best of the best
        runautopilots.save_autopilot_list(best_of_best, save_dir)
    # http://localhost:8787/status
    print("Done")
