import time
import etils

import jax
import mujoco
from mujoco import mjx
import tqdm
import pandas as pd

print("Loading model")
#model = mujoco.MjModel.from_xml_path('ant.xml')
path = etils.epath.Path(etils.epath.resource_path('mujoco')) / ('mjx/test_data/humanoid')
model = mujoco.MjModel.from_xml_path((path / 'humanoid.xml').as_posix())

mjx_model = mjx.device_put(model)

@jax.vmap
def init_model(_):
    return mjx.make_data(mjx_model)

@jax.vmap
def step_model(data):
    return mjx.step(mjx_model, data)

compiled_step_model = jax.jit(step_model)

def time_mjx(batch_size, n_sim_steps=100):
    data_arr = jax.jit(init_model)(jax.numpy.arange(batch_size))
    data_arr = compiled_step_model(data_arr)
    print("Compiling...")
    s_time = time.time()
    for t in range(n_sim_steps):
        compiled_step_model(data_arr)
    running_time = time.time()-s_time
    steps_per_s = n_sim_steps * batch_size / running_time
    print('Num workers: {}'.format(batch_size))
    print('Time: {}'.format(running_time))
    print('Steps / s: {}'.format(steps_per_s))
    return (batch_size, running_time, steps_per_s)

if __name__ == '__main__':
    results = []
    for batch_size in (1,2,4,8,16,32,64,128,256,512,1024,2048,4096,4096*2):
        results.append(time_mjx(batch_size))
    results_df = pd.DataFrame(results, columns=["batch_size", "wallclock_time", "steps_per_second"])
    results_df['simulator'] = 'mjx'
    results_df['parallellisation'] = 'none'
    results_df['model'] = 'humanoid_mjx'
    results_df['parition'] = 'gpu_test'
    results_df['control_inputs'] = 'none'
    results_df['rendering'] = 'none'
    results_df.to_csv('results/mjx_humanoid_mjx.csv')
