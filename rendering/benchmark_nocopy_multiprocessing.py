import time
import multiprocessing

import numpy as np
import tqdm

import mujoco

class DummyRenderer(mujoco.Renderer):

    def makeGLContextCurrent(self):
        if self._gl_context is None:
          raise RuntimeError('render cannot be called after close.')
        self._gl_context.make_current()
    
    def renderDummy(self):       
        # Render scene without any copying
        mujoco._render.mjr_render(self._rect, self._scene, self._mjr_context)


model_filename = '../rodent.xml'
model = mujoco.MjModel.from_xml_path(model_filename)
res = 256

def process_initializer():
    global model
    global data
    global renderer
    global scene_option
    global camera
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
    model.vis.global_.offheight = res
    model.vis.global_.offwidth = res
    camera = "egocentric"
    data = mujoco.MjData(model)
    renderer = DummyRenderer(model, height=res, width=res)

def call_render(_):
    global data
    global model
    global renderer
    global scene_option
    global camera
    renderer.makeGLContextCurrent()
    for i in range(1000):
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        renderer.renderDummy()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    n_workers = 16
    with multiprocessing.Pool(n_workers, process_initializer) as p:
        for t in tqdm.trange(32):
            res = p.map(call_render, np.arange(n_workers))
