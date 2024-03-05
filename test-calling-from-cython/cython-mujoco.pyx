# distutils: include_dirs = mujoco-3.1.2/include/

from libc.stdlib cimport malloc
from libc.string cimport strcpy
from cython.operator cimport dereference
from cython.parallel import prange

#cimport numpy as cnp

cdef extern from "mujoco/mujoco.h" nogil:
    ctypedef double mjtNum
    ctypedef struct mjModel:
        int nq
        int nv
        int nu
    ctypedef struct mjData:
        mjtNum* qpos
    ctypedef struct mjVFS:
        pass
    mjModel* mj_loadXML(const char* filename, const mjVFS* vfs, char* error, int error_sz);
    mjData* mj_makeData(const mjModel* m)
    void mj_step(const mjModel* m, mjData* d)

cdef class MuJoCoRunner:
    cdef readonly int n_envs
    #cdef readonly int n_threads
    cdef mjModel* model
    cdef mjData** data
    
    def __init__(self, bytes model_filename, int n_envs):
        self.n_envs = n_envs
        self._load_model(model_filename)
        self._create_data()

    cdef void _load_model(self, const char* filename):
        cdef char* error_msg = <char *> malloc(1001 * sizeof(char))
        if not error_msg:
            raise MemoryError("Could not allocate space for error message buffer.")
        strcpy(error_msg, "")
        self.model = mj_loadXML(filename, NULL, error_msg, 1000)
        if not self.model:
            raise ValueError("Failed to load XML model. Error message from MuJoCo: " + error_msg)
   
    cdef void _create_data(self):
        cdef int i
        self.data = <mjData**> malloc(self.n_envs * sizeof(mjData*))
        if not self.data:
            raise MemoryError("Could not allocate space for model data.")
        for i in range(self.n_envs):
            self.data[i] = mj_makeData(self.model)
            if not self.data[i]:
                raise MemoryError("Could not allocate space for env number {}.".format(<object>i))
    
    def nq(self):
        return dereference(self.model).nq
        
    def step_singlethreaded(self):
        cdef int i
        for i in range(self.n_envs):
            mj_step(self.model, self.data[i])
            
    def step_multithreaded(self):
        cdef int i
        for i in prange(self.n_envs, nogil=True):
            mj_step(self.model, self.data[i])
            
    def write_qpos_to_array(self, double[:, :] buffer):
        cdef int i
        cdef int j
        cdef int nq = dereference(self.model).nq
        for i in range(self.n_envs):
            for j in range(nq):
                buffer[i, j] = dereference(self.data[i]).qpos[j]

