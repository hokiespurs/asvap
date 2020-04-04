from dask.distributed import Client
from autopilot import autopilot
from simulator import mission
import numpy as np
import time
import runautopilots


# TODO add genetic algorithm code
SERIES = False  # really just for benchmarking
DEBUG = False
DEBUG_NUMS = 4653
CHUNK_SIZE = 500
NUM_PER_WORKER = 50
TOTAL_RUN = int(1e4)
NUM_BEST = 10
RAND_SEED = 1
MAX_SEED = int(1e9)
SAVENAME = "./data/batchruns/AP_30s30s30s_first_tests.txt"

if __name__ == "__main__":
    MISSION_NAME = "./data/missions/increasingangle.txt"
    # class parameters for simulation
    class_params = {
        "boat_params": {},
        "mission_params": {"survey_line_filename": MISSION_NAME, "flip_x": False},
        "environment_params": {},
        "fitness_params": {"gate_length": 1, "offline_importance": 0.8},
        "display_params": {},
    }
    # simulation parameters
    simulation_params = {
        "timestep": 1,
        "num_substeps": 5,
        "do_visual": False,
        "visual_timestep": 0.001,
        "cutoff_max_time": 1000,
        "cutoff_time_gates_same_line": 10,
        "cutoff_time_gates_different_line": 30,
        "cutoff_thresh": [[5, 0.1]],
    }
    # autopilot parameters
    autopilot_params = {
        "num_neurons": [30, 30, 30],
        "rand_weights_method": "randn",  # "rand","randn","randpm"
        "rand_weights_scalar": 1,
        "rand_biases_method": "randn",  # "rand","randn","randpm","zero"
        "rand_biases_scalar": 1,
    }

    # initialize the random seed generator
    random_generator = np.random.default_rng(RAND_SEED)
    all_seeds = random_generator.choice(MAX_SEED, TOTAL_RUN, replace=False)
    my_seed_generator = list(runautopilots.chunks(all_seeds, CHUNK_SIZE))
    # initialize best list
    best_list = runautopilots.reset_best_simulations(5)
    # if parallel, initialize client
    if not SERIES:
        client = Client()
        ncores = sum(client.ncores().values())
        nthreads = sum(client.nthreads().values())
        runtype = f"Parallel with [Cores:{ncores:.0f}] x [Threads:{nthreads:.0f}]"
    else:
        runtype = "Series"

    # If not debugging
    if not DEBUG:
        t_start = time.time()
        # initialize autopilot list
        my_mission = mission.mission(**class_params["mission_params"])
        all_autopilot_list = []
        for seed in range(CHUNK_SIZE):
            all_autopilot_list.append(
                autopilot.ap_nn(
                    my_mission.survey_lines, rand_seed=seed, **autopilot_params
                )
            )
        num_processed = 0
        num_batches = len(my_seed_generator)
        for i, batch_seeds in enumerate(my_seed_generator):
            # process the autopilot list
            t_batch_start = time.time()

            # update autopilots with new random seeds
            for ap, seed in zip(all_autopilot_list, batch_seeds):
                ap.new_random_seed(seed)

            if SERIES:  # RUN IN SERIES
                new_runs = runautopilots.run_autopilots_series(
                    class_params, simulation_params, all_autopilot_list
                )
            else:  # PARALLEL
                new_runs = runautopilots.run_autopilots_parallel(
                    class_params,
                    simulation_params,
                    all_autopilot_list,
                    client,
                    num_per_worker=NUM_PER_WORKER,
                )
            # Print final best runs to screen
            best_list = runautopilots.update_best_simulations(new_runs, best_list)
            runautopilots.print_best_runs(best_list)
            runautopilots.print_best_runs(best_list, SAVENAME)
            print("")
            print(runtype)
            print(
                f"{CHUNK_SIZE:,.0f} element chunk # {i+1}/{num_batches}"
                + f": {runautopilots.timer_str(t_batch_start,time.time())}"
                + f"  ({(time.time()-t_batch_start)/CHUNK_SIZE*1000:.2f}ms/ per boat)"
            )

    else:
        all_autopilot_list = []
        for seed in range(DEBUG_NUMS):
            all_autopilot_list.append(
                autopilot.ap_nn(
                    my_mission.survey_lines, rand_seed=seed, **autopilot_params
                )
            )
        new_runs = runautopilots.debug_autopilot(
            class_params, simulation_params, all_autopilot_list
        )

    # Print final best runs to screen
    best_list = runautopilots.update_best_simulations(new_runs, best_list)
    runautopilots.print_best_runs(best_list)

    print("")
    print(
        f"{TOTAL_RUN:,.0f} in {runautopilots.timer_str(t_start,time.time())}"
        + f"  ({(time.time()-t_start)/TOTAL_RUN*1000:.2f}ms/ per boat)"
    )

    # http://localhost:8787/status
