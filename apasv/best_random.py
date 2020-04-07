from dask.distributed import Client
from autopilot import autopilot
from simulator import mission
import numpy as np
import time
import runautopilots

SERIES = False  # really just for benchmarking
DEBUG = False
DEBUG_NUMS = [782001020]
CHUNK_SIZE = 10000
NUM_PER_WORKER = 50
TOTAL_RUN = int(1e8)
NUM_BEST = 10
RAND_SEED = 14
MAX_SEED = int(1e9)
SAVE_FOLDER = "./data/batchruns/AP_30s30s_line_tests"

if __name__ == "__main__":
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
        "autopilot_type": "apnn",
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

    # initialize the random seed generator
    random_generator = np.random.default_rng(RAND_SEED)
    all_seeds = random_generator.choice(MAX_SEED, TOTAL_RUN, replace=False)
    my_seed_generator = list(runautopilots.chunks(all_seeds, CHUNK_SIZE))
    ndebug = len(DEBUG_NUMS)
    my_seed_generator[0][0:ndebug] = DEBUG_NUMS

    # initialize best list
    best_list = runautopilots.reset_best_simulations(NUM_BEST)
    # if parallel, initialize client
    if not SERIES:
        client = Client()
        ncores = sum(client.ncores().values())
        nthreads = sum(client.nthreads().values())
        runtype = f"Parallel with [Cores:{ncores:.0f}] x [Threads:{nthreads:.0f}]"
    else:
        runtype = "Series"

    # needed for each autpilot
    my_mission = mission.mission(**class_params["mission_params"])

    t_start = time.time()
    if not DEBUG:

        # initialize autopilot list
        ap_template = autopilot.ap_nn(my_mission.survey_lines, **autopilot_params)

        all_autopilot_list = []
        for _ in range(CHUNK_SIZE):
            all_autopilot_list.append(
                autopilot.ap_nn(my_mission.survey_lines, **autopilot_params)
            )

        num_processed = 0
        num_batches = len(my_seed_generator)
        for i, batch_seeds in enumerate(my_seed_generator):
            # process the autopilot list
            t_batch_start = time.time()

            # all_autopilot_list = runautopilots.change_autopilot_seeds(
            #     all_autopilot_list, batch_seeds
            # )
            all_autopilot_list = batch_seeds
            if SERIES:  # RUN IN SERIES
                new_runs = runautopilots.run_autopilots_series(
                    class_params, simulation_params, all_autopilot_list
                )
            else:  # PARALLEL
                # new runs
                new_runs = runautopilots.run_autopilots_parallel(
                    class_params,
                    simulation_params,
                    all_autopilot_list,
                    client,
                    num_per_worker=NUM_PER_WORKER,
                )

            # Print  best runs to screen
            best_list = runautopilots.update_best_simulations(new_runs, best_list)
            runautopilots.print_best_runs(best_list)
            print("")
            print(runtype)
            print(
                f"{CHUNK_SIZE:,.0f} element chunk # {i+1}/{num_batches}"
                + f": {runautopilots.timer_str(t_batch_start,time.time())}"
                + f"  ({(time.time()-t_batch_start)/CHUNK_SIZE*1000:.2f}ms/ per boat)"
            )
            n_total = (i + 1) / num_batches * TOTAL_RUN
            print(
                f"{n_total:,.0f} in {runautopilots.timer_str(t_start,time.time())}"
                + f"  ({(time.time()-t_start)/n_total*1000:.2f}ms/ per boat)"
            )
            # make autopilots to save
            top_ap = []
            for run_num in best_list:
                ap = autopilot.ap_nn(
                    my_mission.survey_lines,
                    rand_seed=int(run_num["id"]),
                    **autopilot_params,
                )
                top_ap.append(ap)
            runautopilots.save_autopilot_list(top_ap, SAVE_FOLDER)
            runautopilots.print_best_runs(best_list, SAVE_FOLDER + "/top.txt")

    else:
        all_autopilot_list = []
        for seed in DEBUG_NUMS:

            all_autopilot_list.append(
                autopilot.ap_nn(
                    my_mission.survey_lines, rand_seed=seed, **autopilot_params
                )
            )
        new_runs = runautopilots.debug_autopilot(
            class_params, simulation_params, all_autopilot_list
        )

    # Print elapsed time
    print(f"Finished in: {runautopilots.timer_str(t_start,time.time())}")

    # http://localhost:8787/status
