import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as spa
import scipy.linalg as sl

def prob(state):
    L = len(state)
    vs0 = np.array([1,0])
    vs1 = np.array([0,1])
    Mf = np.array([vs0 , vs1])
    v = Mf[:,state[0]]
    for i in range(1,L):
        v = np.kron(v ,Mf[:,state[i]])
    rho = np.outer(v,v)
    return np.diag(rho)



def basis(L, bc, Ktot=None, Ref=None):
        if (bc == 0) and (Ktot is not None):
            print( 'MOMENTUM IS NOT A GOOD QUANTUM NUMBER WHEN OBC ARE IMPOSED!' )
        if ( Ref is not None and Ktot != 0 ):
                print( 'REFLECTION SYMMETRY IMPLEMENTED ONLY IN K = 0 SECTOR!' )


        def basis_check(state_num, L, Ktot):
            #The constraint of the state such as Ktot or parity
            if Ktot is None:
                return -2
            else:
                state_temp = state_num
                for R in range(1,L+1):
                    state_temp = Bit_Rotation_R(state_temp,L,Up_state)
                    if( state_temp < state_num ):
                        return -1
                    elif( state_temp == state_num ):
                        if( Ktot%(L/R) != 0 ):
                            return -1
                        else:
                            return R
                return -1


        def basis_check2(state_num, L, Ref,R):
                state_temp = Bit_Reflection( state_num , L )
                for m in range(R):
                    if( state_temp < state_num ):
                        return -2
                    elif( state_temp == state_num ):
                        if( Ref  == -1 ):
                            return -2
                        else:
                            return m
                    state_temp = Bit_Rotation_R(state_temp,L,Up_state)
                return -1

        def add_boson(v):
            vnew = [ num << 1 for num in v ]   # it means put a zero on the right side
            vnew += [ (num << 1) + 1  for num in v  if ( (num & 1) != 1 )  ]  #it means put a 1 on the right side when the right is not 1   
            return vnew

    #     t1 = time.time()
        idx_dict = { }; R_dic = {}; m_dic = {}
        vtot = [ 0 , 1 ] # initialize
        nsite = 1; dict_len = 0;
        while( nsite < L-1 ): # it's fine when len<L, only need to judge at the final step
            vtot = add_boson(vtot)
            nsite += 1

        #This step is for the final boson of the chain. 
        for num in vtot: 
            #dict for those state end with a 0
            num_temp = (num << 1)
            R_temp = basis_check(num_temp, L, Ktot)
            if R_temp!= -1:
                if( Ktot is not None ):
                    if( Ref is not None ):
                        m_temp = basis_check2(num_temp, L, Ref, R_temp)
                        if( m_temp != - 2):
                            R_dic[num_temp] = R_temp
                            m_dic[num_temp] = m_temp
                            idx_dict[ num_temp ] = dict_len
                            dict_len += 1
                    else:
                        R_dic[num_temp] = R_temp
                        idx_dict[ num_temp ] = dict_len     
                        dict_len += 1 
                else:
                    idx_dict[ num_temp ] = dict_len          
                    dict_len += 1

        for num in vtot:
            #dict for those state end with a 1
            if ((num & 1)!= 1) and ( (bc != 1) or (((num>>(L-2))&1)!=1) ): 
                num_temp = ((num << 1)+1)
                R_temp = basis_check(num_temp, L, Ktot)
                if R_temp!= -1:
                    if( Ktot is not None ):
                        if( Ref is not None ):
                            m_temp = basis_check2(num_temp, L, Ref, R_temp)
                            if( m_temp != - 2):
                                R_dic[num_temp] = R_temp
                                m_dic[num_temp] = m_temp
                                idx_dict[ num_temp ] = dict_len
                                dict_len += 1
                        else:
                            R_dic[num_temp] = R_temp
                            idx_dict[ num_temp ] = dict_len     
                            dict_len += 1 
                    else:
                        idx_dict[ num_temp ] = dict_len          
                        dict_len += 1

    #     print( 'Time to build the constraints basis = ' , time.time() - t1 )
        return idx_dict, dict_len




def prob(state_dec, L):# state is a decimal number

    state = [int(i) for i in list('{0:0b}'.format(state_dec).zfill(L))] 
    L = len(state)
    vs0 = np.array([1,0])
    vs1 = np.array([0,1])
    Mf = np.array([vs0 , vs1])
    v = Mf[:,state[0]]
    for i in range(1,L):
        v = np.kron(v ,Mf[:,state[i]])
    rho = np.outer(v,v)
    return np.diag(rho)



def W_site (index, L):
	sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
	Pjp = np.array( [ [ 0. , 0. ] , [ 0. , 1. ] ] )
	Pjm =  np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )
	Id = np.eye(2)
	if index == 0:
		M = np.kron(sigma_x - Id, Pjm)
		for i in range(2, L-1):
			M = np.kron(M, Id)
		M = np.kron(M, Pjm)
	elif index == 1:
		M = Pjm
		M = np.kron(M, np.kron(sigma_x - Id, Pjm) )
		for i in range(3, L):
			M = np.kron(M, Id)
	elif index > 1 and index < L - 1:
		M = Id
		for i in range(1, index - 1):
			M = np.kron(M, Id )
		M = np.kron(M, np.kron(Pjm, np.kron(sigma_x - Id, Pjm)))
		for i in range(index + 2, L):
			M = np.kron(M, Id)
	elif index == L - 1:
		M = Pjm
		for i in range(1, L-2):
			M = np.kron(M, Id)
		M = np.kron(M, np.kron(Pjm, sigma_x - Id))
	return M

