import numpy as np
from scipy import sparse

###### functions we need to build the Fermion-Boson #######



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



def H2sr(matF_1, matF_2, L):

    H=two_FF(matF_1, matF_2, 1, 2, L)
    for i in range(2,L):
            H += two_FF(matF_1, matF_2, i, i+1, L)
    return H



def H2srdef(matF_1, matF_2, L):

    H=two_FF(matF_1, matF_2, 2, 3, L)
    for i in range(3,L):
            H += two_FF(matF_1, matF_2, i, i+1, L)
    return H


def H3srdef(matF_1, matF_2, L):

    H=two_FF(matF_1, matF_2, 3, 4, L)
    for i in range(4,L):
            H += two_FF(matF_1, matF_2, i, i+1, L)
    return H

def k_left(mat,L):
        return sparse.kron(mat,sparse.eye(2**L) )


def k_right(mat,L):
        return sparse.kron(sparse.eye(2**L),mat )

def Liouvillian(H,Jumps,L,gamma):
        Liouville = np.zeros(2**(4*L),dtype=complex).reshape(2**(2*L),2**(2*L)) 
        Liouville = sparse.csc_matrix(Liouville)
        for J in Jumps: 
                J_dag = J.transpose().conjugate()
                J = sparse.csc_matrix(J)
                J_dag = sparse.csc_matrix(J_dag)
                Liouville += gamma * sparse.kron(J.conjugate(),J)\
                           - 0.5 *gamma* sparse.kron(J.transpose().dot(J.conjugate()),np.eye(2**L) )\
                           - 0.5 *gamma* sparse.kron(np.eye(2**L), J_dag.dot(J) )
        Liouville = Liouville - k_left(1j*H.transpose(),L) +  k_right(1j*H,L )
        return Liouville        
def H1srdef(matF_1, matF_2, L):

    H=two_FF(matF_1, matF_2, 1, L, L) + two_FF(matF_1, matF_2, 1, 2, L)
    return H




def H3srdef1(matF_1, matF_2, L):

    H=two_FF(matF_1, matF_2, 1, L, L) + two_FF(matF_1, matF_2, 1, 2, L) + two_FF(matF_1, matF_2, 2, 3, L)
    return H

#Pauli Operator observables

def pauli_op(sites,L):
 Id = np.array( [ [ 1. , 0. ] , [ 0. , 1. ] ] )
 sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
 sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
 sigma_y = np.array([[0, -1j], [1j, 0]])
 pauli_dict = {
    'X': sigma_x,
    'Y': sigma_y,
    'Z': sigma_z,
    'I': Id
 }
 names=['I','X','Y','Z']
 paulis2=[]
 names_pauli2=[]
 for i in names:
  for j in names:
     names_pauli2.append(i+str(sites[0])+j+str(sites[1]))
     paulis2.append(two_FF(pauli_dict[i],pauli_dict[j],sites[0],sites[1],L))
 return paulis2, names_pauli2







