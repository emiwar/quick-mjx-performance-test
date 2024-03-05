import cython_mujoco
import numpy as np
import tqdm

n_envs = 1024
runner = cython_mujoco.MuJoCoRunner(b"rodent.xml", n_envs)
result = np.zeros((n_envs, runner.nq()))
for t in tqdm.trange(10000):
	#runner.step_singlethreaded()
	runner.step_multithreaded()
	runner.write_qpos_to_array(result)
#print(result)
