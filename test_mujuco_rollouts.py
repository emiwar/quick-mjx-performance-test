import mujoco
import mujoco.rollout
import tqdm
import numpy as np
import pandas as pd
import itertools
import time

model_filename = 'humanoid_mjx.xml'
model = mujoco.MjModel.from_xml_path(model_filename)
data = mujoco.MjData(model)
state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

n_steps_list = np.round(2**np.arange(0, 11.25, 0.25)).astype('int')
n_envs_list = np.round(2**np.arange(0, 11.25, 0.25)).astype('int')

test_combinations = itertools.product(n_steps_list, n_envs_list)
n_test_combinations = len(n_steps_list)*len(n_envs_list)
results = []

for n_steps, n_envs in tqdm.tqdm(test_combinations, total=n_test_combinations):
    mujoco.mj_resetData(model, data)
    initial_state = np.zeros((n_envs, state_size))
    control = np.zeros((n_envs, n_steps, model.nu))
    start_time = time.time()
    state, sensordata = mujoco.rollout.rollout(model, data, initial_state, control)
    end_time = time.time()
    results.append((model_filename, n_steps, n_envs, end_time - start_time))

results_df = pd.DataFrame(results, columns=["model", "n_steps",
                                            "n_envs", "run_time"])
results_df["steps_per_second"] = results_df.eval("n_steps * n_envs / run_time")
results_df.to_csv(f"basic_mujoco_rollout_humanoid.csv")
