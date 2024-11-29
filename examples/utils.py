import numpy as np
import os

CURRENT_DIRECTORY = os.getcwd()
DEFAULT_SAVE_DIR = CURRENT_DIRECTORY + '/tmp'


def save_trajectory(
    xs,
    us,
    com,
    FL_force,
    FR_force,
    RL_force,
    RR_force,
    solve_time,
    FL_trans,
    FR_trans,
    RL_trans,
    RR_trans,
    FL_trans_ref,
    FR_trans_ref,
    RL_trans_ref,
    RR_trans_ref,
    L_measured,
    save_name=None,
    save_dir=DEFAULT_SAVE_DIR,
):
    """
    Saves data to a compressed npz file (binary)
    """
    simu_data = {}
    simu_data["xs"] = xs
    simu_data["us"] = us
    simu_data["com"] = com
    simu_data["FL_force"] = FL_force
    simu_data["FR_force"] = FR_force
    simu_data["RL_force"] = RL_force
    simu_data["RR_force"] = RR_force
    simu_data["FL_trans"] = FL_trans
    simu_data["FR_trans"] = FR_trans
    simu_data["RL_trans"] = RL_trans
    simu_data["RR_trans"] = RR_trans
    simu_data["FL_trans_ref"] = FL_trans_ref
    simu_data["FR_trans_ref"] = FR_trans_ref
    simu_data["RL_trans_ref"] = RL_trans_ref
    simu_data["RR_trans_ref"] = RR_trans_ref
    simu_data["L_measured"] = L_measured
    simu_data["solve_time"] = solve_time
    print("Compressing & saving data...")
    if save_name is None:
        save_name = "sim_data_NO_NAME"
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    save_path = save_dir + "/" + save_name + ".npz"
    np.savez_compressed(save_path, data=simu_data)
    print("Saved data to " + str(save_path) + " !")


def load_data(npz_file):
    """
    Loads a npz archive of sim_data into a dict
    """
    d = np.load(npz_file, allow_pickle=True, encoding="latin1")
    return d["data"][()]

def extract_forces(problem, workspace, id):
    force_FL = np.zeros(3)
    force_FR = np.zeros(3)
    force_RL = np.zeros(3)
    force_RR = np.zeros(3)
    contact_states = [False, False, False, False]

    if problem.stages[id].dynamics.differential_dynamics.constraint_models.__len__() == 2:
        if problem.stages[id].dynamics.differential_dynamics.constraint_models[0].name == 'FL_foot':
            force_FL = workspace.problem_data.stage_data[id].dynamics_data.continuous_data.constraint_datas[0].contact_force.linear
            force_RR = workspace.problem_data.stage_data[id].dynamics_data.continuous_data.constraint_datas[1].contact_force.linear
            contact_states = [True, False, False, True]
        else:
            force_FR = workspace.problem_data.stage_data[id].dynamics_data.continuous_data.constraint_datas[0].contact_force.linear
            force_RL = workspace.problem_data.stage_data[id].dynamics_data.continuous_data.constraint_datas[1].contact_force.angular
            contact_states = [False, True, True, False]
    elif problem.stages[id].dynamics.differential_dynamics.constraint_models.__len__() == 4:
        force_FL = workspace.problem_data.stage_data[id].dynamics_data.continuous_data.constraint_datas[0].contact_force.linear
        force_FR = workspace.problem_data.stage_data[id].dynamics_data.continuous_data.constraint_datas[1].contact_force.linear
        force_RL = workspace.problem_data.stage_data[id].dynamics_data.continuous_data.constraint_datas[2].contact_force.linear
        force_RR = workspace.problem_data.stage_data[id].dynamics_data.continuous_data.constraint_datas[3].contact_force.linear
        contact_states = [True, True, True, True]
    return force_FL, force_FR, force_RL, force_RR, contact_states
