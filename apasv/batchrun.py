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


class batchrun:
    def __init__(
        self,
        batch_mission,
        batch_environment,
        batch_boat,
        batch_fitness,
        do_visual=False,
        visual_timestep=0.1,
        name=None,
        timestep=1,
        num_substeps=10,
        cutoff_max_time=1000,
        cutoff_time_gates_same_line=10,
        cutoff_time_gates_different_line=30,
        cutoff_thresh=None,
        num_best=20,
        save_dir="../data/batchruns/",
    ):
        self.mission = batch_mission
        self.environment = batch_environment
        self.boat = batch_boat
        self.fitness = batch_fitness
        # visual
        self.do_visual = do_visual
        self.visual_timestep = 1
        if self.do_visual is True:
            self.visual = display.display()
        else:
            self.visual = None

        # name and dir for saving results
        self.name = name
        self.save_dir = save_dir

        # movement variables
        self.timestep = timestep
        self.num_substeps = num_substeps

        # cutoff variables for when to stop a simulation
        self.cutoff_max_time = cutoff_max_time
        self.cutoff_time_gates_same_line = cutoff_time_gates_same_line
        self.cutoff_time_gates_different_line = cutoff_time_gates_different_line
        if cutoff_thresh is None:
            cutoff_thresh = [[0, -999]]
        self.cutoff_thresh = cutoff_thresh

        # store best runs
        self.num_best = num_best
        self.reset_best_simulations()  # preallocate best runs

    def run(self, autopilot_list, do_parallel=True):
        """ run each of the list of autopilots """
        t_start_run = time.time()
        # if visual is true, or not going to be parallel
        if self.do_visual or not do_parallel:
            for i, ap in enumerate(autopilot_list):
                # run simulation
                sim_result = self.run_simulation(ap)
                # see if simulation result made it into best_runs
                is_changed = self.update_best_simulations(sim_result)
                if is_changed:
                    self.print_best_runs()
                else:
                    self.print_individual_run(sim_result, docr=False, rank=i)
                if self.do_visual:
                    if not self.visual.running:  # if pygame visual closed
                        break
        # run parallel
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:
                all_sim_results = executor.map(self.run_simulation, autopilot_list)
                for sim_result in all_sim_results:
                    self.update_best_simulations(sim_result)
            self.print_best_runs()

        # return the time it took to run in seconds
        run_duration = time.time() - t_start_run
        return run_duration

    def run_simulation(self, my_autopilot):
        """ creates and runs an individual simulation """
        t_start_simulation = time.time()
        # create the simulator
        # use deepcopy for boat and fitness to protect parallel runs
        my_simulator = simulator.simulator(
            boat=deepcopy(self.boat),
            environment=self.environment,
            visual=self.visual,
            fitness=deepcopy(self.fitness),
            autopilot=my_autopilot,
        )
        # preallocate for the cutoff thresholds
        was_cutoff_checked = [False] * len(self.cutoff_thresh)
        loop_criteria = True
        while loop_criteria:
            # get autopilot info and move the boat
            boat_data = my_simulator.get_boat_data()
            new_throttle = my_autopilot.calc_boat_throttle(boat_data)
            my_simulator.set_boat_control(new_throttle)
            my_simulator.update_boat(self.timestep, self.num_substeps)
            # check if loop_criteria is still valid
            # only check the next criteria if the previous one has been met
            boat_time = my_simulator.boat.time
            is_gates_left = False
            is_made_cutoffs = False
            is_ap_not_complete = False
            is_gate_time_gap_good = False
            is_below_max_time = boat_time < self.cutoff_max_time
            if is_below_max_time:
                is_gates_left = not my_simulator.fitness.mission_complete

            if is_gates_left:
                is_made_cutoffs = self.check_cutoff_thresh(
                    boat_time, was_cutoff_checked, my_simulator.get_fitness
                )

            if is_made_cutoffs:
                is_ap_not_complete = not my_autopilot.mission_complete

            if is_ap_not_complete and my_simulator.fitness.current_gate_num > 0:
                is_gate_time_gap_good = self.check_gate_time_gap(
                    boat_time, my_simulator.fitness
                )
            # if it got this far and is_gate_time_gap_good is True
            # everything else had to be true too
            loop_criteria = is_gate_time_gap_good

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
            "id": my_autopilot.id,
            "fitness": fitness_score,
            "percent_complete": percent_complete,
            "time": boat_time,
            "clock": clock_time,
            "run_time": time_to_run,
            "mean_fitness": mean_fitness_score,
            "mean_offline_fitness": mean_offline_fitness,
            "mean_velocity_fitness": mean_velocity_fitness,
        }

    def check_gate_time_gap(self, boat_time, my_fitness):
        """ Check if gap between gates meets the thresholds """
        if my_fitness.current_gate_num > 0:
            gate_num = my_fitness.current_gate_num
            is_same_line = (
                my_fitness.all_gate[gate_num - 1]["line_num"]
                == my_fitness.all_gate[gate_num]["line_num"]
            )
            t_between_gate = (
                boat_time - my_fitness.all_gate_fitness[gate_num - 1]["time"]
            )
            if is_same_line:
                # 1) between gates on the same line
                if t_between_gate > self.cutoff_time_gates_same_line:
                    return False
            else:
                # 2) between gates on different lines
                if t_between_gate > self.cutoff_time_gates_different_line:
                    return False

        return True

    def check_cutoff_thresh(self, boat_time, was_checked, fitness_fun):
        """ Check if threshhold was met """
        # only compute fitness if it needed to check
        for i, (time_fitness, is_checked) in enumerate(
            zip(self.cutoff_thresh, was_checked)
        ):
            if boat_time > time_fitness[0] and not is_checked:
                was_checked[i] = True
                if fitness_fun() < time_fitness[1]:
                    return False

        return True

    def reset_best_simulations(self):
        self.best_simulations = np.zeros((self.num_best, 9))

    def update_best_simulations(self, new_run, old_bests=None):
        """ update the list of best simulations """
        if old_bests is None:
            old_bests = self.best_simulations
        if new_run["fitness"] > old_bests[-1]["fitness"]:
            old_bests[-1] = new_run
            new_bests = sorted(old_bests, key=lambda i: i["fitness"])
        return

    def print_best_runs(self):
        """ Print best_list to the screen """
        # clear screen
        system("cls")
        # print Name, if it exists
        if self.name is not None:
            print("----------------------------")
            print(f"{self.name}")
            print("----------------------------")

        # print header
        headerstr = (
            "       RANK | FITNESS | PER GATE |  OFF FIT |"
            + "  VEL FIT | % COMPLETE | BOAT TIME |"
            + "         SEED |        DATETIME | CPU TIME"
        )
        print(headerstr)
        for rank, sim_data in enumerate(self.best_simulations):
            self.print_sim_results(sim_data, docr=True, rank=rank)

    @staticmethod
    def print_individual_run(data, docr=True, rank=0):
        """ Print results to screen """
        tstr = datetime.datetime.fromtimestamp(data["clock"]).strftime("%m/%d %I:%M %p")
        print_string = (
            f" {rank+1:10.0f} |"
            + f"{data['fitness']:8.2f} |"
            + f"{data['mean_fitness']:9.2f} |"
            + f"{data['mean_offline_fitness']:9.2f} |"
            + f"{data['mean_velocity_fitness']:9.2f} |"
            + f"{data['percent_complete']:11.1f} |"
            + f"{data['time']:10.1f} |"
            + f"{data['id']:13s} |"
            + f"{tstr:>16s} |"
            + f"{data['run_time']:9.3f}"
        )
        if docr:
            print(print_string)
        else:
            print(print_string, end="\r")


if __name__ == "__main__":
    pass


# TODO make simulation data a dictionary
