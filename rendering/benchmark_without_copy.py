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
res = 64

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
model.vis.global_.offheight = res
model.vis.global_.offwidth = res
camera = "egocentric"
data = mujoco.MjData(model)

with DummyRenderer(model, height=res, width=res) as rend:
    rend.makeGLContextCurrent()
    for t in tqdm.trange(10000, desc="No-copy rendering"):
        #mujoco.mj_step(model, data)
        #mujoco.mj_forward(model, data)
        rend.update_scene(data, camera=camera, scene_option=scene_option)
        rend.renderDummy()
    for t in tqdm.trange(10000, desc="Regular rendering"):
        #mujoco.mj_step(model, data)
        #mujoco.mj_forward(model, data)
        rend.update_scene(data, camera=camera, scene_option=scene_option)
        rend.render()