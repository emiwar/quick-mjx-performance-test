import time

import cython_mujoco
import numpy as np
import pandas as pd

def time_cython_mujoco(batch_size, n_sim_steps=1000):
    runner = cython_mujoco.MuJoCoRunner(b"rodent.xml", batch_size)
    result = np.zeros((batch_size, runner.nq()))
    s_time = time.time()
    for t in range(n_sim_steps):
        runner.step_multithreaded()
        runner.write_qpos_to_array(result)
    running_time = time.time()-s_time
    steps_per_s = n_sim_steps * batch_size / running_time
    print('Num workers: {}'.format(batch_size))
    print('Time: {}'.format(running_time))
    print('Steps / s: {}'.format(steps_per_s))
    return (batch_size, running_time, steps_per_s)

results = []
for n_envs in (1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192):
    results.append(time_cython_mujoco(n_envs))
results_df = pd.DataFrame(results, columns=["batch_size", "wallclock_time", "steps_per_second"])
results_df['simulator'] = 'mujoco_with_custom_bindings'
results_df['parallellisation'] = 'none'
results_df['model'] = 'rodent_mjx_version'
results_df['parition'] = 'olveczky'
results_df['control_inputs'] = 'none'
results_df['rendering'] = 'none'
results_df['solver'] = 'default'
results_df['n_cores'] = 32
results_df.to_csv('results/cython_mujoco_rodent_32_cores.csv')
