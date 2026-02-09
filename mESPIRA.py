from numpy import pi,exp,isfinite,round,prod,result_type,empty,ix_,array,arange,argmax,append,hstack
from numpy import zeros,eye,diag,delete,einsum,argmin,vstack,min
from numpy.linalg import svd,lstsq,solve
from scipy.linalg import eig


# Utility Routines
def FC(w,freq,P = 1):
    '''Returns an executable function, which computes Fourier coefficients
    of the SOE.

    Input:
      - w    : 1D-numpy.ndarray, containing the weights of the SOE
      - freq : 1D/2D-numpy.ndarray, contain the frequencies of the SOE.
               If the SOE is univariate, then the frequencies can be
               given as 1D or 2D array.
      - P    : float>0, represents interval length of the Fourier Transform.

    Output:
      -  fc  : Function which compute the Fourier coefficents.
    '''
    if len(freq.shape)==1:
        d = 1
        freq = freq[:,None]
    else :
        d= freq.shape[1]
    
    a = w/(2j*pi)**d*prod(1-exp(P*freq),axis=1)
    b = freq*P/2j/pi

    def fc(k):
        '''Computes the Fourier coefficients of a SOE. If the corresponding SOE is
        univariate, then the imput is just a single value. If the SOE is d-variate,
        then the imput needs to be a container consisting of d integers.'''
        if d ==1:
            return sum(a/(k-b.squeeze()))
        return a@prod(k-b,axis=1)**-1
    return fc

def __cartesian_product(*arrays):
    la = len(arrays)
    dtype = result_type(*arrays)
    arr = empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def FC_arr(w,freq,N,P=1):
    '''Given the weigs,frequencies (of a d-variate SOE) and periodicity P, 
    this functions computes all Fourier coefficients in [-N,N]^d and stores
    them in an appropriate array, which is used for mESPIRA.'''
    if len(freq.shape)==1:
        d = 1
        freq = freq[:,None]
    else :
        d= freq.shape[1]
    fc = FC(w,freq,P = P)
    return array([fc(k) for k in  __cartesian_product(*(d*[arange(-N,N+1)]))]).reshape(d*[2*N+1])


################################################
#main
def __AAA_part(F,K,prec=1e-10,Mmax=None,LP = 0,pinfo=0,P=1):
    if type(Mmax)==type(None):
        Mmax = K.shape[0]//2
    N = len(K)
    F0 = F.copy()
    Z  = K.copy()
    z  = array([]).astype(complex)
    f  = array([]).astype(complex)
    R  = abs(F)
    for _ in range(Mmax+1):
        k1 = argmax(R)
        if R[k1]<prec:
            break
        z  = append(z,Z[k1])
        f  = append(f,F[k1])
        Z  = delete(Z,k1)
        F  = delete(F,k1)
        if LP <2:
            C = (Z[:,None]-z)**-1
            A = F[:,None]*C-C*f
            w = svd(A,False)[-1][-1].conjugate()
            N = C@(w*f)
            D = C@w
            R = abs(F- N/D)

        else:
            M     = len(z)
            L0    = (F[:,None]-f)/(Z[:,None]-z)
            L1    = ((Z*F)[:,None]-z*f)/(Z[:,None]-z)
            L     = hstack([L0,L1])
            _,D,W = svd(L,False)
            W0    = W[:M,:M]
            W1    = W[:M,M:2*M]
            b     = eig(W1,W0)[0]
            a     = lstsq(1/(K[:,None]-b),F0,rcond=None)[0]
            R     = abs(F-(1/(Z[:,None]-b))@a) 

        if pinfo:
            print(max(R))
    
    if LP ==0:
        n = len(z)
        E = zeros((n+1,n+1)).astype(complex)
        E[1:,0] = 1
        E = E+ diag([0]+list(z))
        E[0,1:] = w
        B = eye(n+1)
        B[0,0] = 0
        b = eig(E,B)[0]
        b = b[isfinite(b)]

    elif LP==1:
        Z     = append(Z,z[-1])
        F     = append(F,f[-1])
        z     = z[:-1]
        f     = f[:-1]
        M     = len(z)
        L0    = (F[:,None]-f)/(Z[:,None]-z)
        L1    = ((Z*F)[:,None]-z*f)/(Z[:,None]-z)
        L     = hstack([L0,L1])
        _,D,W = svd(L,False)
        W0    = W[:M,:M]
        W1    = W[:M,M:2*M]
        b     = eig(W1,W0)[0]
        #b     = np.linalg.eig(np.linalg.pinv(W0)@W1)[0]
        #b     = np.linalg.eig(np.linalg.solve(W0,W1))[0]
       
        

    if pinfo:
        print(round(b*2j*pi/P,4))
        print('------')

    
    a     = lstsq(1/(K[:,None]-b),F0,rcond=None)[0]
    return a,b.tolist()

def __LP_part(coef):
    N  = len(coef)//2
    k  = arange(-N,N+1)
    k1 = arange(-N,N+1,2)
    k2 = arange(-N+1,N+1,2)

    L0 = (coef[::2][:,None]-coef[1::2])/(k1[:,None]-k2)
    L1 = ((k*coef)[::2][:,None]-(k*coef)[1::2])/(k1[:,None]-k2)


    L  = hstack([L0,L1])
    _,D,W = svd(L,full_matrices=False)
    M = sum(D>1e-12)

    return eig(solve(W[:M,:M],W[:M,N:N+M]))[0]
def __LP_part(coef):
    N  = len(coef)//2
    k  = arange(-N,N+1)
    k1 = arange(-N,0)
    k2 = arange(1,N+1)

    c1 = coef[:N]
    c2 = coef[N+1:]

    L0 = (c1[:,None]-c2)/(k1[:,None]-k2)
    L1 = ((k1*c1)[:,None]-(k2*c2))/(k1[:,None]-k2)


    L  = hstack([L0,L1])
    _,D,W = svd(L,full_matrices=False)
    M = sum(D>1e-12)

    return eig(solve(W[:M,:M],W[:M,N:N+M]))[0]

def __LP_part(coef):
    N  = len(coef)//2
    k  = arange(-N,N+1)
    k1 = k[N-N//2-1:-(N-N//2)]
    k2 = array(list(k[:N-N//2-1])+list(k[-(N-N//2):]))

    # print(k1)
    # print(k2)



    c1 = coef[N-N//2-1:-(N-N//2)]
    c2 = array(list(coef[:N-N//2-1])+list(coef[-(N-N//2):]))



    L0 = (c1[:,None]-c2)/(k1[:,None]-k2)
    L1 = ((k1*c1)[:,None]-(k2*c2))/(k1[:,None]-k2)


    L  = hstack([L0,L1])
    _,D,W = svd(L,full_matrices=False)
    M = sum(D>1e-12)

    return eig(solve(W[:M,:M],W[:M,N:N+M]))[0]

def __rec_part(fc,prec=1e-10,Mmax=None,LP = 0,pinfo=0,P=1):
    if len(fc.shape)==1:
        d = 1
    else:
        d = len(fc.shape)
    

    N  = fc.shape[0]
    K  = arange(-N//2+1,N//2+1)


    if d==1:
        if LP==3:
            return __LP_part(fc.squeeze())
        else:
            return __AAA_part(fc.squeeze(),K,prec,Mmax,LP=LP,pinfo=pinfo,P=P)[1]
    
    if type(Mmax)==type(None):
        Mmax = N//2

    if LP == 3:
        b = __LP_part(fc[:,*((d-1)*[N//2])])
    else:
        b  = __AAA_part(fc[:,*((d-1)*[N//2])],K,prec,Mmax,LP = LP,pinfo=pinfo,P=P)[1]


    B = []
    C = 1/(K[:,None] -b)
    U,S,V = svd(C,full_matrices=0)
    P = (U@diag(S**-1)@V).T.conj()

    #just don't touch it!
    fc1 = einsum('ji,i...->j...',P,fc)
    ##
    
    def tmp(b1):
        if type(b1)==list:
            return b1
        return [b1]

    for j in range(len(b)):  
        B += [ [b[j]] + tmp(b1) for b1 in __rec_part(fc1[j],prec,Mmax-len(b)+1,pinfo=pinfo)]
    return B



def rESPIRA(fc,Mmax=None,prec=1e-10,P=1,LP=0,pinfo=False):
    '''
    Given the Fourier Coefficients (as well as the corresponding periodicity length), this
    routine reconstructs the weights and requencies of an multivariate SOE.

    Note that the Fourier coefficients (fc) need to satisfy a special format. Let the wanted
    SOE f be d-variate, then fc[k_1,...,k_d] has to coincide with the Fourier coefficient 
    c(f)[k_1,...,k_d].

    Inpute:
      - fc    : d-dimensional numpy.ndarray, contains the Fourier coefficients.
      - Mmax  : integer (default None), sets an upperbound for the wanted SOE order.
      - prec  : float>0, determines the accuracy level for the underlying AAA routine.
      - P     : float>0, sets the perioticity length corresponding to the Fourier coefficients.
      - LP    : 0,1 or 2: "stands for Loewener Pencil", it determines which solver/minimizer
                within the AAA routine is used. For LP=0 (default), the classical AAA routine
                is used, for LP=1 the LP approach is only used in order to determine the poles
                in the final iteration, for LP=2 the LP approach is also used in order to solve
                the minimization problem within the AAA routine.
      - pinfo : bool, (print info) if true then additional information regarding the iterative 
                AAA error is printed.
                
    Output:
     - weig   : 1D-numpy.ndarray, containing the reconstructed weights.
     - freq   : dD-numpy.ndarray, containing the reconstructed weights.'''
    N = len(fc)//2
    K = arange(2*N+1)
    d = len(fc.squeeze().shape)
        

    B = __rec_part(fc,prec,Mmax,LP = LP,pinfo=pinfo,P=P)
    if d>1:
        #B = encode_pole_list(B)
        B = array(B)
    else:
        B = array(B)[:,None]

    K1 = __cartesian_product(*(d*[K]))

    r = array([fc[*k] for k in K1])

    A = array([prod(k-N-B,1) for k in K1])**-1
    a = lstsq(A,r,rcond=None)[0]
    freq = 2j*pi*B/P
    weig = a*(2j*pi)**d/prod(1-exp(freq*P),1)

    return weig,freq


def iESPIRA(fc,tau,prec=1e-12,P=1,LP=0,pinfo=0):
    d = len(fc.shape)
    N = fc.shape[0]//2
    A = []
    B = []
    K = []

    for k in range(d):
        K  += [tuple(k*[N]+[slice(0,2*N+1,1)]+(d-1-k)*[N])]
        F   = fc[K[-1]]
        a,b = __AAA_part(F,arange(-N,N+1),prec,pinfo=pinfo,LP=LP,P=P)
        A += [a]
        B += [array(b)]
        if k==0:
            M = len(a)
        if len(a)!= M:
            print('Problem while computing individual frequencies.')
            return
        else:
            M = len(a)

    if 2*N+1-2*tau<2*M:
        print('Not enough diagonal coefficients given')
        return
    for k in range(d-1):
        F = array([fc[tuple(k*[N]+[l]+[l+2*tau]+(d-2-k)*[N])] for l in range(2*N+1-2*tau)])
        
        b = hstack([B[k],B[k+1]-2*tau])
        c = lstsq(1/(arange(-N,N+1-2*tau)[:,None]-b),F,rcond=None)[0]
        #c,d = AAA_part(F,np.arange(-2*N+tau,2*N+1-tau),prec,pinfo=pinfo)
        c1 = c[:M]
        c2 = c[M:]

        I = []
        for j in range(M):
            J = arange(M)[abs(c1[j]+c2)/abs(c1[j])<1e-5]
            if len(J)==0:
                J = arange(M)
                
            i = argmin(abs(A[k][j]-(c1[j]*B[k+1][J]+c2[J]*B[k][j])/B[k+1][J]))
            I +=[J[i]]
        B[k+1]=B[k+1][I]
        A[k+1]=A[k+1][I]
    
    B = array(B).T
    freq = 2j*pi*B/P

    #K1 = __cartesian_product(*(d*[arange(-N,N+1)]))
    #print(K)
    #print(K1.shape)

    #K1 = vstack([K1[k] for k in K])
    K1 = hstack([hstack([k*[zeros(2*N+1)]+[arange(-N,N+1)]+(d-k-1)*[zeros(2*N+1)]]) for k in range(d)]).T
    F1 = hstack([fc[k] for k in K])
    a  = lstsq(array([prod(1/(k-B),1) for k in K1]),F1,rcond=None)[0]
    
    weig = a*(2j*pi)**d/prod(1-exp(freq*P),1)


    return weig,freq
