import runautopilots

# SAVE_FOLDER = "./data/batchruns/second"
# SAVE_FOLDER = "./data/batchruns/AP_30s30s30s_first_tests"
SAVE_FOLDER = "./data/batchruns/genetic_test2"

autopilot_list = runautopilots.load_autopilot_list(SAVE_FOLDER)

MISSION_NAME = "./data/missions/straightout.txt"
# autopilot parameters
autopilot_params = {
    "num_neurons": [30, 30, 30],
    "rand_weights_method": "randn",  # "rand","randn","randpm"
    "rand_weights_scalar": 1,
    "rand_biases_method": "randn",  # "rand","randn","randpm","zero"
    "rand_biases_scalar": 1,
}
# class parameters for simulation
class_params = {
    "boat_params": {},
    "mission_params": {"survey_line_filename": MISSION_NAME, "flip_x": False},
    "environment_params": {"currents_data": "test"},
    "fitness_params": {"gate_length": 1, "offline_importance": 0.8},
    "display_params": {},
    "autopilot_params": autopilot_params,
    "autopilot_type": "apnn",
}
# simulation parameters
simulation_params = {
    "timestep": 1,
    "num_substeps": 5,
    "do_visual": True,
    "visual_timestep": 0.001,
    "cutoff_max_time": 2000,
    "cutoff_time_gates_same_line": 15,
    "cutoff_time_gates_different_line": 30,
    "cutoff_thresh": [[5, 0.1], [100, 0.1]],
}
# turn partials on
for ap in autopilot_list:
    ap.do_partials = True
# debug simulations
new_runs = runautopilots.debug_autopilot(
    class_params, simulation_params, autopilot_list
)
