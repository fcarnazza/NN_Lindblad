import os
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as spa
import scipy.linalg as sl
import time
import sys
from ED import *
import matplotlib.pyplot as plt
from math import factorial as fac

def binomial(x, y):
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom


#
#	**********  BASIC MATRIX DEFINITIONS  **********
#
sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
N = np.array( [ [ 0. , 0. ] , [ 0. , 1. ] ] )
sigma_plus = np.array( [ [ 0. , 1. ] , [ 0. , 0. ] ] )
sigma_minus = np.array( [ [ 0. , 0. ] , [ 1. , 0. ] ] )
sigma_z = np.array( [ [ -1. , 0. ] , [ 0. , 1. ] ] )
Pj = np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )



L = 20
#U = -0.5

bc = 1
mat2_dis = 2
#state = '0000000000'
#state = '000000000000000000000000'
state = '01'
for i in range(L//2-1):
    state += '01'
           

        
dt = 0.05
T = 10
nstep = int( T/dt )

print('Dynamics')
O1, idx_dict = H_Sym( [sigma_x, sigma_z], range(L) ,mat2_dis, L, bc = bc , output_dict = True )
state_dict = dict((v,k) for k,v in idx_dict.items())


X = O1[0]
Z = O1[1] 

n = [ ]


Hf =  X  

index = idx_dict[ int(state,2) ]
       
psi_in = np.zeros( len(idx_dict) )

psi_in[ index ] = 1


for i in range(len(state_dict)):
    psi = np.zeros( len(idx_dict) )
    psi[i] = 1
    n.append(psi.T.conjugate().dot( Z.dot( psi ) ).real/L )
n = np.array(n)
print(n)
print( 'Time = %.2f ' % (0) )
temp = [ 0 ]
meds = [ psi_in.T.conjugate().dot( Z.dot( psi_in ) ).real/L ]
diagonals = np.multiply(psi_in.conjugate(), psi_in)
d = diagonals.flatten()
print(np.sum(d))
meds1 =  [ np.dot(n, diagonals.flatten())] 
print(meds1)
for i in range( 1, nstep ):

    temp += [ i * dt ]
    psi_in = spa.expm_multiply( - 1j * Hf * dt , psi_in )
    diagonals = np.multiply(psi_in.conjugate(), psi_in)
    print(np.sum(diagonals))
    med = psi_in.T.conjugate().dot( Z.dot( psi_in ) ).real/L
    med1 = np.dot(n, diagonals.flatten())
    meds += [med]
    meds1 += [med1]

    print( 'Time = %.2f ' % (i * dt) , '   ' , med, '   ', med1 )



#np.savetxt('L=%s.txt' % L , np.array([temp,meds]).T )
plt.plot( temp , meds , '-' , label = 'L=%s' % L )
plt.plot(temp, meds1, '-.')


#plt.legend()
#plt.savefig('plot_U=%s.pdf' % U )
plt.show()
    
