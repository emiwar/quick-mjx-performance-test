import os
import mujoco
import mujoco.rollout
import tqdm
import numpy as np
import pandas as pd
import itertools
import time

def benchmark_rendering(model_filename, camera, res, steps, backend="egl",
                        visJoint=False, visSkin=False,
                        visFlexSkin=False, visTexture=True,
                        visFog=False, visShadow=False, visReflection=False):
        os.environ['MUJOCO_GL'] = backend
        model = mujoco.MjModel.from_xml_path(model_filename)
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = visJoint
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = visSkin
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = visFlexSkin
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = visTexture
        model.vis.global_.offheight = res
        model.vis.global_.offwidth = res
        data = mujoco.MjData(model)
        with mujoco.Renderer(model, height=res, width=res) as rend:
            mujoco.mj_step(model, data)
            #mujoco.mj_forward(model, data)
            img = rend.render()
            s_time = time.time()
            for i in range(steps):
                rend.update_scene(data, camera=camera, scene_option=scene_option)
                rend.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = visFog
                rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = visShadow
                rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = visReflection
                rend.render(out=img)
            e_time = time.time()
        return steps / (e_time - s_time)

model_filename = '../rodent.xml'

setups = [
    dict(model_filename=model_filename, camera="side", res=256, steps=1024*8, backend="egl"),
    dict(model_filename=model_filename, camera="egocentric", res=256, steps=1024*8, backend="egl"),
    dict(model_filename=model_filename, camera="egocentric", res=64, steps=1024*8, backend="egl"),
]

features = [
    dict(visJoint=True,  visSkin=True, visFlexSkin=True, visTexture=True, visFog=True, visShadow=True, visReflection=True),
    dict(visJoint=False, visSkin=True, visFlexSkin=True, visTexture=True, visFog=True, visShadow=True, visReflection=True),
    dict(visJoint=False, visSkin=True, visFlexSkin=False, visTexture=True, visFog=False, visShadow=True, visReflection=True),
    dict(visJoint=False, visSkin=False, visFlexSkin=False, visTexture=True, visFog=False, visShadow=True, visReflection=True),
    dict(visJoint=False,  visSkin=False, visFlexSkin=False, visTexture=True, visFog=True, visShadow=True, visReflection=True),
    dict(visJoint=False, visSkin=False, visFlexSkin=False, visTexture=True, visFog=False, visShadow=False, visReflection=True),
    dict(visJoint=False, visSkin=False, visFlexSkin=False, visTexture=True, visFog=False, visShadow=True, visReflection=False),
    dict(visJoint=False, visSkin=False, visFlexSkin=False, visTexture=True, visFog=False, visShadow=False, visReflection=False),
    dict(visJoint=False, visSkin=False, visFlexSkin=False, visTexture=False, visFog=True, visShadow=True, visReflection=True),
    dict(visJoint=False, visSkin=True, visFlexSkin=False, visTexture=True, visFog=False, visShadow=False, visReflection=False),
    dict(visJoint=False, visSkin=True, visFlexSkin=False, visTexture=False, visFog=False, visShadow=False, visReflection=False),
    dict(visJoint=False, visSkin=True, visFlexSkin=True, visTexture=True, visFog=False, visShadow=False, visReflection=False),
    dict(visJoint=False, visSkin=True, visFlexSkin=True, visTexture=False, visFog=False, visShadow=False, visReflection=False),
    dict(visJoint=False, visSkin=True, visFlexSkin=True, visTexture=False, visFog=True, visShadow=False, visReflection=False),
    dict(visJoint=False, visSkin=False, visFlexSkin=False, visTexture=False, visFog=True, visShadow=False, visReflection=False),
    dict(visJoint=True, visSkin=False, visFlexSkin=False, visTexture=False, visFog=False, visShadow=False, visReflection=False),
    dict(visJoint=False, visSkin=False, visFlexSkin=False, visTexture=False, visFog=False, visShadow=False, visReflection=False),
]

results = []
for setup in setups:
    camera = setup["camera"]
    res = setup["res"]
    for feature_set in tqdm.tqdm(features, desc=f"{camera}, {res}x{res}"):
        setup.update(feature_set)
        result = setup.copy()
        result["fps"] = benchmark_rendering(**setup)
        results.append(result)
pd.DataFrame(results).to_csv("results/feature_benchmarking_olveczkygpu.csv")

