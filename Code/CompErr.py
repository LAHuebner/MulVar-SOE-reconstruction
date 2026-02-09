from itertools import permutations
from numpy import result_type,empty,ix_
from numpy import arange,max,argmin,exp,array,linspace

def compare(w1,f1,w2,f2,nodes=10000):
    '''Computes the relative approximation error for the weigths, frequencies and function values.
    Input:
    ------
      - w1    : 1D numpy.ndarray, weigths of first sum of exponentials
      - f1    : 2D numpy.ndarray, frequency matrix of first sum of exponentials
      - w2    : 1D numpy.ndarray, weights of second sum of exponentials
      - f2    : 2D numpy.ndarray, frequency matrix of second sum of exponentials
      - nodes : integer, number of nodes used for estimating error corresponding to 
                the function values.
    Output:
    -------
      None, the results will be printed.
    '''
    #### Need to sort frequencies first
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = result_type(*arrays)
        arr = empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)
    
    d = f1.shape[1]
    
    I = list(set(permutations(arange(len(f1)))))
    R = []
    for i in I:
        R+= [max(abs(f1-f2[array(i)]))]
    I = array(I[argmin(R)])

    f2 = f2[I]
    w2 = w2[I]
    #### End sorting
    
    try:
        print('err(freq)       \t: ',max([max(abs((f1-f2).T[d])/max(abs(f1).T[d])) for d in range(d)]))
        print('err(weig)       \t: ',max(abs(w1-w2))/max(abs(w1)))
    except:
        print('Failure to find correct frequencies / weights.')
    
    
    SOE1 = lambda x : w1@exp(f1@x)
    SOE2 = lambda x : w2@exp(f2@x)
    d = f1.shape[1]

    #X = cartesian_product(*(d*[np.linspace(-10,10,51)]))
    X = cartesian_product(*(d*[linspace(-10,10,int(nodes**(1/d)))]))
    tmp1 = array([SOE1(x) for x in X])
    tmp2 = array([SOE2(x) for x in X])
    print(f'err([-{10},{10}]^{d})    \t: ',max(abs(tmp1-tmp2))/max(abs(tmp1)))