import multiprocessing
import time
import etils

import mujoco
import pandas as pd

def run_steps(worker_id):
    m = mujoco.MjModel.from_xml_path('rodent.xml')
    d = mujoco.MjData(m)
    for i in range(100_000):
        #d.ctrl = np.random.normal(0, 0.01, m.nu)
        mujoco.mj_step(m, d)
    return True

def time_multicore(n_workers=64):
    s_time = time.time()
    with multiprocessing.Pool(n_workers) as p:
        p.map(run_steps, range(n_workers))
    running_time = time.time()-s_time
    steps_per_s = 100_000 * n_workers / running_time
    print('Num workers: {}'.format(n_workers))
    print('Time: {}'.format(running_time))
    print('Steps / s: {}'.format(steps_per_s))
    return (n_workers, running_time, steps_per_s)
    
if __name__ == '__main__':
    results = []
    for n_workers in (1,2,4,8,16,32,64,128):
        results.append(time_multicore(n_workers))
    results_df = pd.DataFrame(results, columns=["n_workers", "wallclock_time", "steps_per_second"])
    results_df['simulator'] = 'mujoco'
    results_df['parallellisation'] = 'multiprocessing'
    results_df['model'] = 'rodent_mjx_optimized'
    results_df['parition'] = 'olveczky'
    results_df['control_inputs'] = 'none'
    results_df['rendering'] = 'none'
    results_df.to_csv('results/multiprocessing_rodent_mjx_48_cores.csv')
