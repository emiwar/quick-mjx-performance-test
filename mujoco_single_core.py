import mujoco
import tqdm

m = mujoco.MjModel.from_xml_path('rodent.xml')
d = mujoco.MjData(m)

for i in tqdm.trange(200000):
    #d.ctrl = np.random.normal(0, 0.01, m.nu)
    mujoco.mj_step(m, d)
