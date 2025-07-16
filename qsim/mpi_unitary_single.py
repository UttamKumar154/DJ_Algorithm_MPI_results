from mpi4py import MPI
import numpy as np
from math import *
import sys
import ast
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
    
def single_rot(gate, param, err_param, state, q, num_qubits):
    
    if rank == 0:
        c = err_param[0]*np.cos(param + err_param[1])
        s = err_param[0]*np.sin(param + err_param[1])
        if str(gate) == 'rz':
            k = [1, 2]
        elif str(gate) == 'rx':
            k = [2, 3]
        elif str(gate) == 'ry':
            k = [3, 1]
        temp = [c, s]
        lt, mt, rt = 4**q, 4, 4**(num_qubits-q-1)
        state = np.reshape(state, (lt, mt, rt))
        temp1 = state[:, k[0], :].copy()
        temp2 = state[:, k[1], :].copy()
    
    else:
        temp, temp1, temp2 = None, None, None
    
    temp = comm.bcast(temp, root=0)
    temp1 = comm.bcast(temp1, root=0)
    temp2 = comm.bcast(temp2, root=0)
    
    if rank == 0:
        var = None
        N = 1
    else:
        N = 1
        
        idx = (rank-1)%N
        if rank <= N:
            var = (temp1[idx::N]) * temp[0]
        elif N < rank <= 2*N:
            var = (temp2[idx::N]) * temp[1]
        elif 2*N < rank <= 3*N:
            var = (temp2[idx::N]) * temp[0]
        else:
            var = (temp1[idx::N]) * temp[1]
    data = comm.gather(var, root=0)
    
    if rank == 0:
        for t in range(N):
            state[:, k[0], :][t::N] = data[t+1] - data[t+1+N]
            state[:, k[1], :][t::N] = data[t+1+2*N] + data[t+1+3*N]
        return state

if __name__ == "__main__":
    gate = (sys.argv[1])
    param = float(sys.argv[2])
    err_param = ast.literal_eval(sys.argv[3])
    input_file = sys.argv[4]
    output_file = sys.argv[5]
    q = int(sys.argv[6])
    num_qubits = int(sys.argv[7])
    # state = ast.literal_eval(sys.argv[4])
    # q = int(sys.argv[5])
    # n = int(sys.argv[6])
    # Load input state
    state = np.load(input_file)

    result = single_rot(gate, param, err_param, state, q, num_qubits)

    if rank == 0:
        np.save(output_file, result)

    # result1 = single_rot(gate, param, err_param, state, q, n)
    # if rank == 0:
    #     print(repr(result1))
        