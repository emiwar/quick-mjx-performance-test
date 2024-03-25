import os
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

print(os.environ['MUJOCO_GL'])
renderer = mujoco.Renderer(model, height=64, width=64)
mujoco.mj_step(model, data)
def update_and_render(data):
    renderer.update_scene(data)
    renderer.render()

for t in tqdm.trange(10000):
    update_and_render(data)
