import threading
import concurrent.futures
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

n_workers = 20
n_steps_list = np.round(2**np.arange(0, 10.25, 0.25)).astype('int')
n_envs_list = np.round(2**np.arange(0, 10.25, 0.25)).astype('int')

test_combinations = itertools.product(n_steps_list, n_envs_list)
n_test_combinations = len(n_steps_list)*len(n_envs_list)
results = []

thread_local = threading.local()
def thread_initializer():
    thread_local.data = mujoco.MjData(model)

def call_rollout(initial_state, control, state, sensordata):
    mujoco.rollout.rollout(model, thread_local.data, initial_state, control,
                           skip_checks=True, nroll=initial_state.shape[0],
                           nstep=state.shape[1], state=state, sensordata=sensordata)

for n_steps, n_envs in tqdm.tqdm(test_combinations, total=n_test_combinations):
    mujoco.mj_resetData(model, data)
    initial_state = np.zeros((n_envs, state_size))
    control = np.zeros((n_envs, n_steps, model.nu))
    state = np.empty((n_envs, n_steps, state_size))
    sensordata = np.empty((n_envs, n_steps, model.nsensordata))
    
    n = n_steps // n_workers
    chunks = []  # a list of tuples, one per worker
    for i in range(n_workers-1):
        chunks.append((initial_state[i*n:(i+1)*n],
                       control[i*n:(i+1)*n],
                       state[i*n:(i+1)*n],
                       sensordata[i*n:(i+1)*n]))
    # last chunk, absorbing the remainder:
    chunks.append((initial_state[(n_workers-1)*n:],
                   control[(n_workers-1)*n:],
                   state[(n_workers-1)*n:],
                   sensordata[(n_workers-1)*n:]))
    

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=n_workers, initializer=thread_initializer) as executor:
        start_time = time.time()
        futures = []
        for chunk in chunks:
            futures.append(executor.submit(call_rollout, *chunk))
        for future in concurrent.futures.as_completed(futures):
            future.result()
        end_time = time.time()
    results.append((model_filename, n_steps, n_envs, n_workers, end_time - start_time))

results_df = pd.DataFrame(results, columns=["model", "n_steps",
                                            "n_envs", "n_workers", "run_time"])
results_df["steps_per_second"] = results_df.eval("n_steps * n_envs / run_time")
results_df.to_csv(f"results/multithreaded_mujoco_rollout_humanoid.csv")
