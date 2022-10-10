import numpy as np
import scipy.sparse as sps

def in_state(sp): 
#input list or array, sp entries 0 or 1 (number of fermions per site), allowed states are ...010000... 01100... , ...00001 not contempleted
    L=len(sp)
    #spins
    v_up = np.array([1,0])
    v_dn = np.array([0,1]) 
    M_sp = np.array([v_dn, v_up])
    v = M_sp[:,sp[0]] 
    for i in range(1,L):
        v = sps.kron(v , M_sp[:,sp[i]])
    return np.squeeze(v.toarray())

#### HAMILTONIAN TERMS ####
# GLOBAL

def one_Op(mat, position, L): ###One-body operator only fermion!###
    
    ####check the first position####
    if position == 1:
        H = sps.csc_matrix(mat)
        for i in range(2,L+1):
            H = sps.kron( H , sps.eye(2))
    else:
        H = sps.eye(2)
        for i in range(2,L+1):
            if i != position:
                H = sps.kron( H , sps.eye(2))
            elif i == position:
                H = sps.kron( H , sps.csc_matrix(mat))
    return H


def two_Op(mat_F1, mat_F2, position1, position2, L): #### Two-bodies fermionic operators, pos1<pos2 ALWAYS!####
    
    if position1 == 1:
        H = sps.csc_matrix(mat_F1)
        for i in range(2, L+1):
            if i != position2:
                H = sps.kron( H , sps.eye(2))
            elif i == position2:
                H = sps.kron( H , sps.csc_matrix(mat_F2))
    else:
        H = sps.eye(2)
        for i in range(2, L+1):
            if i == position1:
                H = sps.kron( H , sps.csc_matrix(mat_F1))
            elif i+1 == position2:
                H = sps.kron( H , sps.csc_matrix(mat_F2))
            else:
                H = sps.kron( H , sps.eye(2) )
    return H


def Dis_loc(jump1, jump2, position, L):

    if position == 1:
        D1 = sps.csc_matrix(jump1)
        D2 = sps.csc_matrix(jump2)
        for i in range(2, L+1):
            D1 = sps.kron(D1, sps.eye(2))
            D2 = sps.kron(D2, sps.eye(2))
    else :
        D1 = sps.eye(2)
        D2 = sps.eye(2)
        for i in range(2, L+1):
            if i == position:
                D1 = sps.kron(D1, sps.csc_matrix(jump1))
                D2 = sps.kron(D2, sps.csc_matrix(jump2))
            else:
                D1 = sps.kron(D1, sps.eye(2))
                D2 = sps.kron(D2, sps.eye(2))

    D = sps.kron(D1, D1) - 1/2*(sps.kron(D2, sps.eye(2**L)) + sps.kron(sps.eye(2**L), D2))
    return D 


#################################################### GLOBAL OPERATORS ####################################

def H1f(mat_F, L):
    H = one_Op(mat_F, 1, L)
    for i in range(2, L+1):
        H += one_Op(mat_F, i, L)
    return H

def H2f(mat_F1, mat_F2, L):
    H = np.zeros((2**L, 2**L))
    H = sps.csc_matrix(H)
    for i in range(1, L):
        for j in range(i, L):
            H += two_Op(mat_F1, mat_F2, i, j,  L)
    return H

def Diss(jump1, jump2, L):
    Dis = Dis_loc(jump1, jump2, 1, L)
    for i in range(2, L+1):
        Dis += Dis_loc(jump1, jump2, i, L)
    return Dis

