from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([Extension("cython_mujoco", ["cython-mujoco.pyx"],
                            libraries=["mujoco"],
                            library_dirs=["mujoco-3.1.2/lib"],
                            extra_compile_args=['-fopenmp'],
                            extra_link_args=['-fopenmp']
                            )])
)

