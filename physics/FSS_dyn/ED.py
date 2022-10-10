import numpy as np
import scipy.sparse as ssp


dim = 2
#	**********  BIT OPERATION  **********
#
def Bit_Rotation_R(num, bits, Up_State):
    num &= Up_State
    bit = num & 1
    num >>= 1
    if(bit):
        num |= (1 << (bits-1))

    return num

def Bit_Rotation_L(num, bits):
    bit = num & (1 << (bits-1))
    num <<= 1
    if(bit):
        num |= 1
    num &= (( 1 << bits )-1)
    return num


def Bit_Reflection(num, bits):
    result = 0
    for i in range(bits):
        if (num >> i) & 1: result |= 1 << (bits - 1 - i)
    return result

def Bit_Get(num,i):
    return (num >> i) & 1


def Bit_Set(num,i,val):
    if( val == 0 ):
        return num & ~(1<<i)
    else:
        return num | (1<<i)




def one( i , row_idx, full_idx , mat,  a_list , row_idxs , column_idxs , entries , idx_dict  ):

    a1 = Bit_Get( full_idx, i )

    for a2 in a_list[a1]:
        old_col_idx = Bit_Set( full_idx , i , a2 )

        col_idx = idx_dict.get( old_col_idx )

        if( col_idx is not None ):
            row_idxs += [row_idx]
            column_idxs += [col_idx]
            entries += [ mat[a1,a2] ]

def one_mom( i , row_idx, full_idx , mat,  a_list , row_idxs , column_idxs , entries, idx_dict , Ktot, R_dic ):

    a1 = Bit_Get( full_idx, i )

    for a2 in a_list[a1]:

        if( a1 == a2 ):
            old_col_idx = full_idx
            row_idxs += [row_idx]
            col_idx = idx_dict.get( old_col_idx )
            column_idxs += [col_idx]
            entries += [ mat[a1,a2] ]
        else:
            t_idx = Bit_Set( full_idx , i , a2 )
            rep_idx = t_idx
            l_fin = 0
            for l in range(1,L):
                t_idx = Bit_Rotation_R(t_idx,L,Up_state)
                if( t_idx < rep_idx ):
                    l_fin = l
                    rep_idx = t_idx
            col_idx = idx_dict.get( rep_idx )
            if( col_idx is not None ):
                row_idxs += [row_idx]
                column_idxs += [col_idx]
                if( Ktot == 0 ):
                    entries += [ np.sqrt(1.*R_dic[full_idx]/R_dic[rep_idx]) * mat[a1,a2]  ]
                else:
                    entries += [ np.sqrt(1.*R_dic[full_idx]/R_dic[rep_idx]) * mat[a1,a2] * np.exp(1j*l_fin*2*np.pi*Ktot/L) ]


def one_ref( i , row_idx, full_idx , mat,  a_list , row_idxs , column_idxs , entries, idx_dict , Ktot, Ref, R_dic, m_dic ):

    a1 = Bit_Get( full_idx, i )

    for a2 in a_list[a1]:

        if( a1 == a2 ):
            old_col_idx = full_idx
            row_idxs += [row_idx]
            col_idx = idx_dict.get( old_col_idx )
            column_idxs += [col_idx]
            entries += [ mat[a1,a2] ]
        else:
            t_idx = Bit_Set( full_idx , i , a2 )
            rep_idx = t_idx
            l_fin = 0
            q_fin = 0
            for l in range(1,L):
                t_idx = Bit_Rotation_R(t_idx,L,Up_state)
                if( t_idx < rep_idx ):
                    l_fin = l
                    rep_idx = t_idx

            t_idx = Bit_Reflection(t_idx,L)
            if( t_idx < rep_idx ):
                l_fin = 0
                q_fin = 1
                rep_idx = t_idx

            for l in range(1,L):
                t_idx = Bit_Rotation_R(t_idx,L,Up_state)
                if( t_idx < rep_idx ):
                    l_fin = l
                    q_fin = 1
                    rep_idx = t_idx


            col_idx = idx_dict.get( rep_idx )
            if( col_idx is not None ):
                row_idxs += [row_idx]
                column_idxs += [col_idx]
                m_row = m_dic[full_idx]
                m_col = m_dic[rep_idx]
                if( m_row != -1 ):
                    R_row = R_dic[full_idx] / 2.
                else:
                    R_row = R_dic[full_idx]

                if( m_col != -1 ):
                    R_col = R_dic[rep_idx] / 2.
                else:
                    R_col = R_dic[rep_idx]


                entries += [ Ref**q_fin  * mat[a1,a2] * np.sqrt(1.*R_row/R_col)  ]



def two( i , j , row_idx, full_idx , mat, ab_list , row_idxs , column_idxs , entries  , idx_dict ):

    if( i > j ):
        return two( j , i , row_idx, full_idx , mat, ab_list , row_idxs , column_idxs , entries , idx_dict )

    a1 = Bit_Get( full_idx, i )
    b1 = Bit_Get( full_idx, j )

    for a2,b2 in ab_list[a1,b1]:
        old_col_idx = Bit_Set( full_idx , i , a2 )
        old_col_idx = Bit_Set( old_col_idx , j , b2 )

        col_idx = idx_dict.get( old_col_idx )
        if( col_idx is not None ):
            row_idxs += [row_idx]
            column_idxs += [col_idx]
            entries += [ mat[a1,a2,b1,b2] ]

def two_mom( i , j , row_idx, full_idx , mat, ab_list , row_idxs , column_idxs , entries, idx_dict , Ktot, R_dic ):

    if( i > j ):
        return two_mom( j , i , row_idx, full_idx , mat, ab_list , row_idxs , column_idxs , entries, idx_dict , Ktot,  R_dic )

    a1 = Bit_Get( full_idx, i )
    b1 = Bit_Get( full_idx, j )

    for a2,b2 in ab_list[a1,b1]:

        if( a1 == a2 and b1 == b2 ):
            old_col_idx = full_idx
            row_idxs += [row_idx]
            col_idx = idx_dict.get( old_col_idx )
            column_idxs += [col_idx]
            entries += [ mat[a1,a2,b1,b2] ]
        else:
            t_idx = Bit_Set( full_idx , i , a2 )
            t_idx = Bit_Set( t_idx , j , b2 )
            rep_idx = t_idx
            l_fin = 0
            for l in range(1,L):
                t_idx = Bit_Rotation_R(t_idx,L,Up_state)
                if( t_idx < rep_idx ):
                    l_fin = l
                    rep_idx = t_idx
            col_idx = idx_dict.get( rep_idx )
            if( col_idx is not None ):
                row_idxs += [row_idx]
                column_idxs += [col_idx]
                entries += [ np.sqrt(1.*R_dic[full_idx]/R_dic[rep_idx]) * mat[a1,a2,b1,b2] * np.exp(1j*l_fin*2*np.pi*Ktot/L) ]


def two_ref( i , j , row_idx, full_idx , mat, ab_list , row_idxs , column_idxs , entries, idx_dict , Ktot, Ref, R_dic, m_dic ):

    if( i > j ):
        return two_ref( j , i ,  row_idx, full_idx , mat, ab_list , row_idxs , column_idxs , entries, idx_dict , Ktot, Ref, R_dic, m_dic )

    a1 = Bit_Get( full_idx, i )
    b1 = Bit_Get( full_idx, j )

    for a2,b2 in ab_list[a1,b1]:

        if( a1 == a2 and b1 == b2 ):
            old_col_idx = full_idx
            row_idxs += [row_idx]
            col_idx = idx_dict.get( old_col_idx )
            column_idxs += [col_idx]
            entries += [ mat[a1,a2,b1,b2] ]
        else:
            t_idx = Bit_Set( full_idx , i , a2 )
            t_idx = Bit_Set( t_idx , j , b2 )
            rep_idx = t_idx
            l_fin = 0
            q_fin = 0
            for l in range(1,L):
                t_idx = Bit_Rotation_R(t_idx,L,Up_state)
                if( t_idx < rep_idx ):
                    l_fin = l
                    rep_idx = t_idx

            t_idx = Bit_Reflection(t_idx,L)
            if( t_idx < rep_idx ):
                l_fin = 0
                q_fin = 1
                rep_idx = t_idx

            for l in range(1,L):
                t_idx = Bit_Rotation_R(t_idx,L,Up_state)
                if( t_idx < rep_idx ):
                    l_fin = l
                    q_fin = 1
                    rep_idx = t_idx


            col_idx = idx_dict.get( rep_idx )
            if( col_idx is not None ):
                row_idxs += [row_idx]
                column_idxs += [col_idx]
                m_row = m_dic[full_idx]
                m_col = m_dic[rep_idx]
                if( m_row != -1 ):
                    R_row = R_dic[full_idx] / 2.
                else:
                    R_row = R_dic[full_idx]

                if( m_col != -1 ):
                    R_col = R_dic[rep_idx] / 2.
                else:
                    R_col = R_dic[rep_idx]

                entries += [ Ref**q_fin  * mat[a1,a2] * np.sqrt(1.*R_row/R_col)  ]


def three( i , j , k , row_idx, full_idx , mat, abc_list , row_idxs , column_idxs , entries  , idx_dict ):

    if( i < j < k ):
            a1 = Bit_Get( full_idx, i )
            b1 = Bit_Get( full_idx, j )
            c1 = Bit_Get( full_idx, k )

            for a2,b2,c2 in abc_list[a1,b1,c1]:
                old_col_idx = Bit_Set( full_idx , i , a2 )
                old_col_idx = Bit_Set( old_col_idx , j , b2 )
                old_col_idx = Bit_Set( old_col_idx , k , c2 )

                col_idx = idx_dict.get( old_col_idx )
                if( col_idx is not None ):
                    row_idxs += [row_idx]
                    column_idxs += [col_idx]
                    entries += [ mat[a1,a2,b1,b2,c1,c2] ]
    else:
        i,j,k = np.sort( [i,j,k] )
        return three( i , j , k , row_idx, full_idx , mat, abc_list , row_idxs , column_idxs , entries , idx_dict )











### bc = 0 OBC
### bc = 1 PBC
### Ktot = 0 , .. , L - 1 
### Ref = 1,-1 (solo se Ktot = 0)


### H_Sym( [N,sigmaZ] , [N] , [ [ N , N ]  , [ sigmax,sigmax ] ] , 2 , 34 , 1 )


#def H_Sym( matrices1_list, matrices1_stag_list, mat2_dis, L, bc, Ktot=None , Ref=None,  output_dict = False ):
def H_Sym( matrices1_list, i1_list ,mat2_dis, L, bc, Ktot=None , Ref=None,  output_dict = False ):
    #
    #	**********  FUNCTION FOR BUILDING BASIS  **********
    # this is the only function need change for other symmetry.


    Up_State = (2**L - 1) ###############Will be useful in Rotation_R




    print( 'bc = ' , bc )

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
        return idx_dict, dict_len, R_dic, m_dic
    
    if( bc == 0 and Ktot != None ):
        print( 'MOMENTUM IS NOT A GOOD QUANTUM NUMBER WHEN OBC ARE IMPOSED!' )



    idx_dict, nrow, R_dic, m_dic = basis(L, bc, Ktot, Ref)
    ncolumn = nrow




    #i1_list = range(L) # the site that the operators need apply
    H1s = []
    for matA in matrices1_list:

        a_list = np.empty(dim,dtype=object)
        for a1 in range(dim):
            a_list[a1] = []
            for a2 in range(dim):
                if( np.abs( matA[a1,a2] ) != 0.0 ):
                    a_list[a1] += [a2] 



        entries = []; row_idxs = []; column_idxs = []

        if( Ktot is not None ): #  for translation invariance
            if( Ref is None ):
                for full_idx in idx_dict:
                    row_idx = idx_dict[full_idx]
              
                    for i in i1_list:
                        one_mom( i , row_idx, full_idx , matA, a_list , row_idxs , column_idxs , entries, idx_dict , Ktot, R_dic )
            else:
                for full_idx in idx_dict:
                    row_idx = idx_dict[full_idx]
              
                    for i in i1_list:
                        one_ref( i , row_idx, full_idx , matA, a_list , row_idxs , column_idxs , entries, idx_dict , Ktot, Ref, R_dic, m_dic )

        else:
            for full_idx in idx_dict:
                row_idx = idx_dict[full_idx]
        
                for i in i1_list:  #### SE VUOI SOLO N_2, metti for i in [2]
                    one( i , row_idx, full_idx ,matA, a_list , row_idxs , column_idxs , entries , idx_dict )

        if( ( Ktot is None ) or ( Ktot == 0 ) ):
            H1s += [ssp.coo_matrix( (entries,(row_idxs,column_idxs)) , (nrow,nrow) , dtype=np.float)]
        else:
            H1s += [ssp.coo_matrix( (entries,(row_idxs,column_idxs)) , (nrow,nrow) , dtype=np.cfloat)]
        entries = []; row_idxs = []; column_idxs = []

        if( Ktot is not None ): #  for translation invariance
            if( Ref is None ):
                for full_idx in idx_dict:
                    row_idx = idx_dict[full_idx]
              
                    for i in i1_list:
                        one_mom( i , row_idx, full_idx , (-1)**i * matA, a_list , row_idxs , column_idxs , entries, idx_dict , Ktot, R_dic )
            else:
                for full_idx in idx_dict:
                    row_idx = idx_dict[full_idx]
              
                    for i in i1_list:
                        one_ref( i , row_idx, full_idx , (-1)**i * matA, a_list , row_idxs , column_idxs , entries, idx_dict , Ktot, Ref, R_dic, m_dic )

        else:
            for full_idx in idx_dict:
                row_idx = idx_dict[full_idx]
        
                for i in i1_list:
                    one( i , row_idx, full_idx , (-1)**i * matA, a_list , row_idxs , column_idxs , entries , idx_dict )


    if output_dict == False:
        return H1s
    else:
        if( Ktot is not None ):
            if( Ref is not None):
                return H1s, idx_dict, R_dic, m_dic
            else:
                return H1s,idx_dict, R_dic
        else:
            return H1s, idx_dict
        

###############################################################################################################

def Full_Basis(L,bc):

    def add_boson(v):
        vnew = [ num << 1 for num in v ]   # it means put a zero on the right side
        vnew += [ (num << 1) + 1  for num in v  if ( (num & 1) != 1 )  ]  #it means put a 1 on the right side when the right is not 1   
        return vnew

    idx_dict = { };
    vtot = [ 0 , 1 ] # initialize
    nsite = 1; dict_len = 0;
    while( nsite < L-1 ): # it's fine when len<L, only need to judge at the final step
        vtot = add_boson(vtot)
        nsite += 1

    #This step is for the final boson of the chain. 
    for num in vtot: 
        #dict for those state end with a 0
        num_temp = (num << 1)
        idx_dict[ num_temp ] = dict_len          
        dict_len += 1

    for num in vtot:
        #dict for those state end with a 1
        if ((num & 1)!= 1) and ( (bc != 1) or (((num>>(L-2))&1)!=1) ): 
            num_temp = ((num << 1)+1)
            idx_dict[ num_temp ] = dict_len          
            dict_len += 1

    #     print( 'Time to build the constraints basis = ' , time.time() - t1 )
    return idx_dict



def Ext_Dict(R_dic,m_dic,idx_dict,full_idx_dict,L):
    ext_dict = {}
    comp_dict = {}
    for state in idx_dict:
        Ri = R_dic[state]
        if( m_dic[state] != -1 ):
            comp_dict[state] = 1./ np.sqrt(R_dic[state])
            ext_dict[state] = [full_idx_dict[state]]
            idx_list = ext_dict[state]
            for r in range(Ri-1):
                state = Bit_Rotation_R(state,L,Up_state)
                idx_list += [full_idx_dict[state]]
        else:
            comp_dict[state] = 1./ np.sqrt(2*R_dic[state])
            ext_dict[state] = [full_idx_dict[state]]
            idx_list = ext_dict[state]
            for r in range(Ri-1):
                state = Bit_Rotation_R(state,L,Up_state)
                idx_list += [full_idx_dict[state]]
            state = Bit_Reflection(state,L)
            idx_list += [full_idx_dict[state]]
            for r in range(Ri-1):
                state = Bit_Rotation_R(state,L,Up_state)
                idx_list += [full_idx_dict[state]]

    return ext_dict,comp_dict     



def Ext_State(psi,ext_dict,comp_dict,idx_dict,full_dict):
    psi_ex = np.empty( len(full_dict) )
    #print('Here')
    for state in ext_dict:
        #print( state )
        i = idx_dict[state]
        psi_ex[ ext_dict[state] ] = psi[i] * comp_dict[state]

            
    return psi_ex




def One_Site(matA,i,L,idx_dict,Ktot=None,Ref=None,R_dic=None,m_dic=None):

    nrow = len(idx_dict)
    ncolumn = nrow

    a_list = np.empty(dim,dtype=object)
    for a1 in range(dim):
        a_list[a1] = []
        for a2 in range(dim):
            if( np.abs( matA[a1,a2] ) != 0.0 ):
                a_list[a1] += [a2] 


    entries = []; row_idxs = []; column_idxs = []

    if( Ktot is not None ): #  for translation invariance
        if( Ref is None ):
            for full_idx in idx_dict:
                row_idx = idx_dict[full_idx]
          
                one_mom( i , row_idx, full_idx , matA, a_list , row_idxs , column_idxs , entries, idx_dict , Ktot, R_dic )
        else:
            for full_idx in idx_dict:
                row_idx = idx_dict[full_idx]
          

                one_ref( i , row_idx, full_idx , matA, a_list , row_idxs , column_idxs , entries, idx_dict , Ktot, Ref, R_dic, m_dic )

    else:
        for full_idx in idx_dict:
            row_idx = idx_dict[full_idx]

            one( i , row_idx, full_idx ,matA, a_list , row_idxs , column_idxs , entries , idx_dict )

    if( ( Ktot is None ) or ( Ktot == 0 ) ):
        return ssp.coo_matrix( (entries,(row_idxs,column_idxs)) , (nrow,nrow) , dtype=np.float)
    else:
        return ssp.coo_matrix( (entries,(row_idxs,column_idxs)) , (nrow,nrow) , dtype=np.cfloat)

############################# FUNCTION FOR RESTORING THE DENSITY MATRIX######################################

def diag_from_state(v, dictio ,L):

    def diag_1(sp): 
        L=len(sp)
        vs0 = np.array([1,0])
        vs1 = np.array([0,1])
        Mf = np.array([vs0 , vs1])
        v = Mf[:,sp[0]]    
        for i in range(1,L):
            v = ssp.kron(v , Mf[sp[i]])
        diag = v.multiply(v.conjugate())
        return diag
    
    get_bin = lambda x, L: format(x, 'b').zfill(L)
    diagn = ssp.csr_matrix(np.zeros(2**L))
    for i in range(len(v)):
        st = [int(d) for d in str(get_bin(dictio[i], L))]
        diagn += (v[i]*v[i].conjugate())*diag_1(st)
    return(diagn)
