from dask.distributed import Client
from autopilot import autopilot
from simulator import mission
import time
import runautopilots


# TODO Implement chunks and updates into the code
# TODO Maybe add neuralnet.change_seed(seed) to NN and ap code
# then you dont have to make a new autopilot each time
# TODO add genetic algorithm code
# TODO see if storm is still crashing, or just on close
SERIES = False
DEBUG = False
DEBUG_NUM = 4653
CHUNK_SIZE = 5000
NUM_PER_WORKER = 50
TOTAL_RUN = int(1e5)

if __name__ == "__main__":
    MISSION_NAME = "./data/missions/increasingangle.txt"
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
    # make autopilot list
    my_mission = mission.mission(**class_params["mission_params"])
    all_autopilot_list = []
    for seed in range(CHUNK_SIZE):
        all_autopilot_list.append(
            autopilot.ap_nn(
                my_mission.survey_lines,
                num_neurons=[30, 30, 30],
                rand_seed=seed,
                rand_weights_method="randn",  # "rand","randn","randpm"
                rand_weights_scalar=1,
                rand_biases_method="randn",  # "rand","randn","randpm","zero"
                rand_biases_scalar=1,
            )
        )

    # process the autopilot list
    t1 = time.time()

    if SERIES:
        new_runs = runautopilots.run_autopilots_series(
            class_params, simulation_params, all_autopilot_list
        )
    elif DEBUG:
        new_runs = runautopilots.debug_autopilot(
            class_params, simulation_params, [all_autopilot_list[DEBUG_NUM]]
        )
    else:  # PARALLEL
        client = Client()
        new_runs = runautopilots.run_autopilots_parallel(
            class_params,
            simulation_params,
            all_autopilot_list,
            client,
            batch_size=NUM_PER_WORKER,
        )
        client.close()
    # print best runs
    best_list = runautopilots.reset_best_simulations(10)
    best_list = runautopilots.update_best_simulations(new_runs, best_list)
    runautopilots.print_best_runs(best_list)

    print("")
    print(time.time() - t1)

    # http://localhost:8787/status
