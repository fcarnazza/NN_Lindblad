import numpy as np
import scipy.linalg as LA

rho_loc=np.array([np.random.normal() for i in range(16)]).reshape(4,4)+1j*np.array([np.random.normal() for i in range(16)]).reshape(4,4)

rho_loc=rho_loc.dot(rho_loc.conjugate().transpose())

rho_loc=rho_loc/rho_loc.trace()

print(LA.expm(1j*rho_loc) )

e, U = LA.eigh(rho_loc)

print((U*np.exp(1j*e))@np.conj(U.T))

