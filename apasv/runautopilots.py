from os import system
import os
from simulator import boat, display, environment, mission, simulator
from autopilot import autopilot
import time
import datetime
from dask.distributed import Client
from functools import partial
import itertools
import pickle
from pathlib import Path
import glob

# sys.path.insert(0, "../autopilot")
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# TODO add framework for "gate check"
# if a boat is not on track to make the top N, stop the simulation


def run_simulator(my_simulator, simulation_params):
    t_start_simulation = time.time()
    was_cutoff_checked = [False] * len(simulation_params["cutoff_thresh"])
    loop_criteria = True

    while loop_criteria:
        # get autopilot info and move the boat
        boat_data = my_simulator.get_boat_data()
        new_throttle = my_simulator.autopilot.calc_boat_throttle(boat_data)
        my_simulator.set_boat_control(new_throttle)
        my_simulator.update_boat(
            simulation_params["timestep"], simulation_params["num_substeps"]
        )
        if my_simulator.visual is not None:
            my_simulator.update_visual(simulation_params["visual_timestep"])
            if not my_simulator.visual.running:
                break
        # check if loop_criteria is still valid
        # only check the next criteria if the previous one has been met
        boat_time = my_simulator.boat.time

        # if boat is beyond cutoff time
        if boat_time > simulation_params["cutoff_max_time"]:
            break

        # if the boat has been through all the gates
        if my_simulator.fitness.mission_complete:
            break

        # if the autopilot thinks its finished the mission
        if my_simulator.autopilot.mission_complete:
            break

        # if the boat missed a hard cutoff threshold
        is_made_cutoffs = check_cutoff_thresh(
            boat_time,
            was_cutoff_checked,
            my_simulator.get_fitness,
            simulation_params["cutoff_thresh"],
        )
        if not is_made_cutoffs:
            break

        is_gate_time_gap_good = check_gate_time_gap(
            boat_time,
            my_simulator.fitness,
            simulation_params["cutoff_time_gates_same_line"],
            simulation_params["cutoff_time_gates_different_line"],
        )

        if not is_gate_time_gap_good:
            break

    # STATS TO RETURN
    # time in seconds to run the simulation
    time_to_run = time.time() - t_start_simulation
    # percent complete
    current_gate = my_simulator.fitness.current_gate_num
    total_gates = len(my_simulator.fitness.all_gate)
    percent_complete = 100 * current_gate / total_gates
    # clock time
    clock_time = time.time()
    # overall fitness
    fitness_score = my_simulator.get_fitness()
    offline_fitness_score = my_simulator.fitness.mean_offline_fitness
    velocity_fitness_score = my_simulator.fitness.mean_velocity_fitness
    # average fitness
    if current_gate == total_gates:
        current_gate -= 1  # otherwise it'd be dividing by too many gates
    mean_fitness_score = fitness_score / (current_gate + 1)
    mean_offline_fitness = offline_fitness_score / (current_gate + 1)
    mean_velocity_fitness = velocity_fitness_score / (current_gate + 1)

    return {
        "id": my_simulator.autopilot.id,
        "fitness": fitness_score,
        "percent_complete": percent_complete,
        "boat_time": boat_time,
        "clock": clock_time,
        "run_time": time_to_run,
        "mean_fitness": mean_fitness_score,
        "mean_offline_fitness": mean_offline_fitness,
        "mean_velocity_fitness": mean_velocity_fitness,
    }


def check_gate_time_gap(boat_time, my_fitness, cutoff_same_line, cutoff_different_line):
    """ Check if gap between gates meets the thresholds """
    gate_num = my_fitness.current_gate_num
    if gate_num > 0 and gate_num < len(my_fitness.all_gate):
        is_same_line = (
            my_fitness.all_gate[gate_num - 1]["line_num"]
            == my_fitness.all_gate[gate_num]["line_num"]
        )
        t_between_gate = boat_time - my_fitness.all_gate_fitness[gate_num - 1]["time"]
        if is_same_line:
            # 1) between gates on the same line
            if t_between_gate > cutoff_same_line:
                return False
        else:
            # 2) between gates on different lines
            if t_between_gate > cutoff_different_line:
                return False

    return True


def check_cutoff_thresh(boat_time, was_checked, fitness_fun, cutoff_thresh):
    """ Check if threshhold was met """
    # only compute fitness if it needed to check
    for i, (time_fitness, is_checked) in enumerate(zip(cutoff_thresh, was_checked)):
        if boat_time > time_fitness[0] and not is_checked:
            was_checked[i] = True
            if fitness_fun() < time_fitness[1]:
                return False

    return True


def chunks(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for first_ind in range(0, len(lst), chunk_size):
        last_ind = first_ind + chunk_size
        yield lst[first_ind:last_ind]


def run_autopilots_parallel(
    class_params, simulation_params, autopilot_list, client, num_per_worker=50
):
    # slice the autopilot_list into smaller lists
    ap_chunks = list(chunks(autopilot_list, num_per_worker))
    run_function = partial(run_autopilots_series, class_params, simulation_params)
    # add each list to the client
    chunk_runs = client.map(run_function, ap_chunks)
    # return all fitness dictionary
    chunk_output = client.gather(chunk_runs)

    return list(itertools.chain.from_iterable(chunk_output))


def run_autopilots_series(class_params, simulation_params, autopilot_list):
    """ Run a list of autopilots in series"""
    # Make the simulator
    my_mission = mission.mission(**class_params["mission_params"])
    my_simulator = simulator.simulator(
        boat=boat.boat(**class_params["boat_params"]),
        environment=environment.environment(**class_params["environment_params"]),
        visual=None,
        fitness=mission.fitness(my_mission, **class_params["fitness_params"]),
        autopilot=autopilot_list[0],
    )
    # if passing in random seeds
    if class_params["autopilot_type"] == "apnn":
        my_ap = autopilot.ap_nn(
            my_mission.survey_lines, **class_params["autopilot_params"],
        )
    # For each autopilot
    all_fitness = []
    for ap in autopilot_list:
        # set autopilot
        if class_params["autopilot_type"] == "apnn":
            my_ap.new_random_seed(ap)
            # x = deepcopy(my_ap)
            # my_ap = autopilot.ap_nn(
            #     my_mission.survey_lines,
            #     rand_seed=ap,
            #     **class_params["autopilot_params"],
            # )
        else:
            my_ap = ap
        my_simulator.autopilot = my_ap
        # reset simulation
        my_simulator.reset()
        # run simulation
        ap_fitness = run_simulator(my_simulator, simulation_params)
        # append fitness list
        all_fitness.append(ap_fitness)

    # return fitness dictionary list
    return all_fitness


def debug_autopilot(class_params, simulation_params, autopilot_list):
    """ Run a list of autopilots in series"""
    # Make the simulator
    my_mission = mission.mission(**class_params["mission_params"])
    my_simulator = simulator.simulator(
        boat=boat.boat(**class_params["boat_params"]),
        environment=environment.environment(**class_params["environment_params"]),
        visual=display.display(**class_params["display_params"]),
        fitness=mission.fitness(my_mission, **class_params["fitness_params"]),
        autopilot=autopilot_list[0],
    )
    # For each autopilot
    all_fitness = []
    for ap in autopilot_list:
        # set autopilot
        my_simulator.autopilot = ap
        # reset simulation
        my_simulator.reset()
        # run simulation
        ap_fitness = run_simulator(my_simulator, simulation_params)
        # append fitness list
        all_fitness.append(ap_fitness)

    # return fitness dictionary list
    return all_fitness


def reset_best_simulations(num_best):
    val = {
        "id": "reset",
        "fitness": 0,
        "percent_complete": 0,
        "boat_time": 0,
        "clock": time.time(),
        "run_time": 0,
        "mean_fitness": 0,
        "mean_offline_fitness": 0,
        "mean_velocity_fitness": 0,
    }

    return [val] * num_best


def update_best_simulations(new_runs, old_bests=None, num_bests=None):
    """ update the list of best simulations """
    # return whether the new run made the list
    if old_bests is None:
        all_runs = new_runs
        if num_bests is None:
            num_bests = 10
    else:
        new_runs_sorted = sorted(new_runs, key=lambda i: i["fitness"])
        num_bests = len(old_bests)
        if len(new_runs_sorted) > num_bests:
            all_runs = new_runs_sorted[-num_bests:] + old_bests
        else:
            all_runs = new_runs_sorted + old_bests

    new_bests = sorted(all_runs, key=lambda i: i["fitness"])
    new_bests.reverse()
    new_bests = new_bests[0:num_bests]

    return new_bests


def print_best_runs(best_runs, filename=None):
    """ Print best_list to the screen """
    headerstr = (
        "       RANK | FITNESS | PER GATE |  OFF FIT |"
        + "  VEL FIT | % COMPLETE | BOAT TIME |"
        + "        DATETIME | CPU TIME |          SEED |"
    )
    if filename is None:
        # clear screen
        system("cls")
        # print header
        print(headerstr)
        file_handle = None
    else:
        file_handle = open(filename, "w+")
        file_handle.write(headerstr + "\n")

    for rank, sim_data in enumerate(best_runs):
        print_individual_run(sim_data, docr=True, rank=rank, file_handle=file_handle)

    if filename is not None:
        file_handle.close()


def print_individual_run(data, docr=True, rank=0, file_handle=None):
    """ Print results to screen """
    tstr = datetime.datetime.fromtimestamp(data["clock"]).strftime("%m/%d %I:%M %p")
    print_string = (
        f" {rank+1:10.0f} |"
        + f"{data['fitness']:8.2f} |"
        + f"{data['mean_fitness']:9.2f} |"
        + f"{data['mean_offline_fitness']:9.2f} |"
        + f"{data['mean_velocity_fitness']:9.2f} |"
        + f"{data['percent_complete']:11.1f} |"
        + f"{data['boat_time']:10.1f} |"
        + f"{tstr:>16s} |"
        + f"{data['run_time']:9.3f} |"
        + f"{data['id']:>14s} |"
    )
    if file_handle is None:
        if docr:
            print(print_string)
        else:
            print(print_string, end="\r")
    else:
        file_handle.write(print_string + "\n")


def timer_str(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def change_autopilot_seeds(autopilot_list, rand_seeds):
    """ change the random seed for a list of autopilots"""
    for ap, seed in zip(autopilot_list, rand_seeds):
        ap.new_random_seed(seed)

    return autopilot_list


def save_autopilot(ap, filename):
    with open(filename, "wb") as output:
        pickle.dump(ap, output, pickle.HIGHEST_PROTOCOL)


def load_autopilot(filename):
    with open(filename, "rb") as input:
        ap = pickle.load(input)
    return ap


def save_autopilot_list(ap_list, save_dir):
    # make save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # save each autopilot
    for i, ap in enumerate(ap_list):
        filename = f"/ap{i:02.0f}.pkl"
        save_autopilot(ap, save_dir + filename)


def load_autopilot_list(folder_name):
    fnames = glob.glob(folder_name + "/*.pkl")
    return [load_autopilot(f) for f in fnames]


if __name__ == "__main__":
    # simulation object parameters
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
    for seed in range(1000):
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
    client = Client()
    t1 = time.time()
    # new_runs = run_autopilots_series(
    #     class_params, simulation_params, all_autopilot_list
    # )

    new_runs = run_autopilots_parallel(
        class_params, simulation_params, all_autopilot_list, client, num_per_worker=50
    )

    best_list = reset_best_simulations(10)
    best_list = update_best_simulations(new_runs, best_list)
    print_best_runs(best_list)
    print("")
    print(time.time() - t1)
