import multiprocessing
import time

import mujoco
import pandas as pd
import tqdm
import numpy as np
import torch

def load_model_to_global(batch_per_worker=32):
    global m
    global d
    m = mujoco.MjModel.from_xml_path('rodent.xml')
    d = [mujoco.MjData(m) for i in range(batch_per_worker)]

def take_step(control_actuations):
    global m
    global d
    batch_size = len(d)
    obs_size = d[0].qpos.size + d[0].qvel.size
    output = np.empty((batch_size, obs_size))
    for i in range(batch_size):
        d[i].ctrl = control_actuations[i]
        mujoco.mj_step(m, d[i])
        qpos = d[i].qpos
        qvel = d[i].qvel
        output[i, :qpos.size] = qpos
        output[i, qpos.size:] = qvel
    return output

def time_multicore(batch_per_worker=32, n_workers=1, control_size=30):
    s_time = time.time()
    with multiprocessing.Pool(n_workers, load_model_to_global, (batch_per_worker,)) as p:
        for t in range(1000):
            input_cuda = 0.01*torch.randn((n_workers, batch_per_worker, control_size), device='cuda')
            control_input = input_cuda.cpu()
            obs = p.map(take_step, control_input)
            obs = np.stack(obs)
            if obs.shape != (n_workers, batch_per_worker, 74+73):
                print(f"ERROR: obs.shape = {obs.shape}")
                break
            output_cuda = torch.from_numpy(obs).cuda()
    running_time = time.time()-s_time
    steps_per_s = 1000 * batch_per_worker * n_workers / running_time
    print('Num workers: {}'.format(n_workers))
    print('Environments per worker: {}'.format(batch_per_worker))
    print('Time: {}'.format(running_time))
    print('Steps / s: {}'.format(steps_per_s))
    return (n_workers, running_time, steps_per_s)
    
if __name__ == '__main__':
    n_workers = 64
    results = []
    for batch_per_worker in (1,2,4,8,16,32,64,128,256,512):
        results.append(time_multicore(batch_per_worker, n_workers=64))
    results_df = pd.DataFrame(results, columns=["batch_per_worker", "wallclock_time", "steps_per_second"])
    results_df['simulator'] = 'mujoco'
    results_df['parallellisation'] = 'synchronized_multiprocessing'
    results_df['model'] = 'virtual_rodent_mjx_version'
    results_df['parition'] = 'local'
    results_df['control_inputs'] = '0.01*torch.randn'
    results_df['rendering'] = 'none'
    results_df.to_csv('results/synchronized_multiprocessing_rodent_mjx.csv')
