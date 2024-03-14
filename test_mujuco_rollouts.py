import mujoco
import mujoco.rollout
import tqdm
import numpy as np
import pandas as pd
import itertools
import time

model_filename = 'rodent.xml'
model = mujoco.MjModel.from_xml_path(model_filename)
data = mujoco.MjData(model)
state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

n_steps_list = np.round(2**np.arange(0, 9.25, 0.25)).astype('int')
n_envs_list = np.round(2**np.arange(0, 9.25, 0.25)).astype('int')

test_combinations = itertools.product(n_steps_list, n_envs_list)
n_test_combinations = len(n_steps_list)*len(n_envs_list)
results = []

for n_steps, n_envs in tqdm.tqdm(test_combinations, total=n_test_combinations):
    mujoco.mj_resetData(model, data)
    #It seems it might be necessary to do a single-step rollout first
    #after resetting to get consistent results?
    mujoco.rollout.rollout(model, data,
                           np.zeros((n_envs,state_size)),
                           np.zeros((n_envs, 1, model.nu)))
    
    initial_state = np.zeros((n_envs, state_size))
    control = np.zeros((n_envs, n_steps, model.nu))
    #Pre-allocate outputs
    state = np.empty((n_envs, n_steps, state_size))
    sensordata = np.empty((n_envs, n_steps, model.nsensordata))
    start_time = time.time()
    mujoco.rollout.rollout(model, data, initial_state, control,
                           state=state, sensordata=sensordata)
    end_time = time.time()
    results.append((model_filename, n_steps, n_envs, end_time - start_time))

results_df = pd.DataFrame(results, columns=["model", "n_steps",
                                            "n_envs", "run_time"])
results_df["steps_per_second"] = results_df.eval("n_steps * n_envs / run_time")
results_df.to_csv(f"results/basic_mujoco_rollout_rodent.csv")
