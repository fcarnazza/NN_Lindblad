import numpy as np
from scipy import sparse

###### functions we need to build the Fermion-Boson #######
###### We fix two bosons per site, LOCAL dimension, dim = 6 #####

########    CONTROLLARE GLI INDICI UNA VOLTA PRIMA DI USARE!!!! ############################

####GENERATE INITIAL WAVE FUNCTION####

def in_state(sp): #input list or array, sp entries 0 or 1 (number of fermions per site), bos entries 0,1,2 number of bosons per site
    L=len(sp)
    vs0 = np.array([1,0])
    vs1 = np.array([0,1])
    Mf = np.array([vs0 , vs1])
    v = Mf[:,sp[0]]     
    for i in range(1,L):
        v = sparse.kron(v ,Mf[:,sp[i]])
    return np.squeeze(v.toarray())

#### LOCAL TERMS ####

def one_F(mat, position, L): ###One-body operator only fermion!###
    
    ####check the first position####
    if position == 1:
        H=sparse.csc_matrix(mat)
        for i in range(2,L+1):
            H = sparse.kron( H , sparse.eye(2))
    else:
        H =  sparse.eye(2) 
        for i in range(2,L+1):
            if i != position:
                H = sparse.kron( H , sparse.eye(2) )
            elif i == position:
                H = sparse.kron( H ,  sparse.csc_matrix(mat)    )
    return H

        

def two_FF(mat_F1, mat_F2, position1, position2, L): #### Two-bodies fermionic operators, pos1<pos2 ALWAYS!####
    
    if position1 == 1:
        H   =  sparse.csc_matrix(mat_F1)
        for i in range(1,L):
            if (i+1) != position2:
                H = sparse.kron( H  ,    sparse.eye(2)  )
            elif (i+1) == position2:
                H = sparse.kron( H ,   sparse.csc_matrix(mat_F2)   )
    else:
        H = sparse.eye(2) 
        for i in range(1,L):
            if (i+1) == position1:
                H = sparse.kron( H ,  sparse.csc_matrix(mat_F1) )
            elif (i+1) == position2:
                H = sparse.kron( H ,   sparse.csc_matrix(mat_F2) )
            #elif (i+1) != position1 & (i+1) != position2:
            else:
                H = sparse.kron( H , sparse.eye(2) )
    return H

###################################################################	 FULL HAMILTONIANS HERE  #############################################################################

def H1f(mat_F, L):
    H = one_F(mat_F, 1, L)
    for i in range(2,L+1):
        H += one_F(mat_F, i, L)
    return H



def H2fflr(matF_1, matF_2, L, alpha):
    H=two_FF(matF_1, matF_2, 1, 2, L)
    for i in range(1,L):
        if i == 1:
            for j in range(i+2,L+1):
                H += (two_FF(matF_1, matF_2, i,j,L))/((j-i)**alpha)
        else:
            for j in range(i+1, L+1):
                H += (two_FF(matF_1, matF_2, i,j,L))/((j-i)**alpha)
    return H
