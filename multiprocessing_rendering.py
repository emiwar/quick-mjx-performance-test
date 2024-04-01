import mujoco
import tqdm
import numpy as np
import pandas as pd
import itertools
import time
import multiprocessing

model_filename = 'rodent.xml'
model = mujoco.MjModel.from_xml_path(model_filename)

def process_initializer():
    global model
    global data
    global renderer
    #global ctx
    data = mujoco.MjData(model)
    #ctx = mujoco.GLContext(64, 64)
    #ctx.make_current()
    renderer = mujoco.Renderer(model, height=64, width=64)
    
def call_render(_):
    global data
    global model
    for i in range(20):
        renderer.update_scene(data)
        renderer.render()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    n_workers = 20
    with multiprocessing.Pool(n_workers, process_initializer) as p:
        for t in tqdm.trange(100):
            res = p.map(call_render, np.arange(n_workers))
        
