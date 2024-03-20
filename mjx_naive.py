import time
import etils
import os

import jax
import mujoco
from mujoco import mjx
import tqdm
import pandas as pd

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

print("Loading model")
model = mujoco.MjModel.from_xml_path('rodent.xml')
#path = etils.epath.Path(etils.epath.resource_path('mujoco')) / ('mjx/test_data/humanoid')
#model = mujoco.MjModel.from_xml_path((path / 'humanoid.xml').as_posix())
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
model.opt.iterations = 1
model.opt.ls_iterations = 4
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
    for batch_size in (1,2,4,8,16,32,64,128,256,512):#,1024,2048,4096,4096*2):
        results.append(time_mjx(batch_size))
    results_df = pd.DataFrame(results, columns=["batch_size", "wallclock_time", "steps_per_second"])
    results_df['simulator'] = 'mjx'
    results_df['parallellisation'] = 'none'
    results_df['model'] = 'rodent_mjx'
    results_df['parition'] = 'olveczkygpu'
    results_df['control_inputs'] = 'none'
    results_df['rendering'] = 'none'
    results_df['solver'] = 'newton'
    results_df.to_csv('results/mjx_newton_solver_rodent_olveckygpu.csv')
