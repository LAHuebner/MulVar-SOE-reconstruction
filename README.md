# MulVar SOE Reconstruction
This GitHub page is intended to store the entire code needed to recreate the results presented in the preprint [Parameter estimation for multivariate exponential sums via iterative rational approximation](https://arxiv.org/abs/2504.19157). It is joint work with [Nadiia Derevianko](https://www.cs.cit.tum.de/sccs/personen/nadiia-derevianko/).

The goal of this project is the reconstruction of multivariate SOEs (sums of exponentials) from their Fourier coefficients. More precisely, fix an SOE

$$f(t_1,\dots,t_d) = \sum_{j=1}^M \gamma_j e^{ t_1 \lambda_{j,1} + \dots+t_d \lambda_{j,d}}, $$

then its Fourier coefficients are given by

$$c_{\mathbf{k}}(f):= \frac{1}{P^d}\int_{[0,P]^d}f(\mathbf t) e^{-\frac{2\pi i}{P} \langle \mathbf k,\mathbf t\rangle}\mathrm d \mathbf{t}=\sum_{j=1}^M\gamma_j \prod_{\ell =1}^d \frac{e^{\lambda_{j,\ell}P}-1}{\lambda_{j,\ell}P-2\pi i k_\ell},\qquad k=(k_1,\dots,k_d) \in\mathbb Z^d.$$

The weights $\gamma_j$ and frequencies $\lambda_{j,k}$ are then reconstructed in terms of a rational reconstruction problem. For more details see the linked preprint.

# How to Use the Code
The folder `Code` consists of three files: `Routines.py`, which stores the reconstruction routines; `CompErr.py`, which contains a routine to compute the relative approximation error of the weights and frequencies; and `Examples.ipynb`, which displays all examples shown in the preprint. In order to use this code, download the folder or the files separately and make sure that `Routines.py` and `CompErr.py` are in the same folder in which your kernel is running. Then the routines can be accessed by the usual import
```python
import numpy as np
from Routines import FC_arr,algorithm3,algorithm4
from CompErr import compare
```

Let us quickly discuss the given routines. First of all, `FC_arr` computes the corresponding Fourier coefficient matrix. It is recommended to use this command to ensure that the format is compatible with the actual recovery routine. Then we have two reconstruction routines `algorithm3` and `algorithm4`, which are defined in the given preprint. The routine `compare` determines the reconstruction errors.



See the following example illustrating how to use the routines:
```python
# Setting weights and frequencies
freq = 1j*np.array([[2.21**0.5,3.33],
                    [-5.63,-5**0.5],
                    [-3.47,6**0.5],
                    [-7.1**0.5,-4.5],
                    [0.46,-9.44]])
weig = np.array([3,2,1,2,1])


# Setting Parameters
P    = 4
N    = 15
tau  = 7
Mmax = None


# Creating Array with Fourier Coefficients
coef = FC_arr(weig,freq,N,P)


# Running Code and compare results
print('Results Algorithm 3')
weig1,freq1 = algorithm3(coef,tau=tau,P = P)
compare(weig,freq,weig1,freq1)
print()

print('Results Algorithm 4')
# Running Code and compare results
weig1,freq1 = algorithm4(coef,Mmax=Mmax,P = P)
compare(weig,freq,weig1,freq1)```

**Output**
```text
Results Algorithm iESPIRA
err(freq)       	:  8.1820e-14
err(weig)       	:  3.2124e-13
err([-10,10]^2)    	:  8.4067e-13

Results Algorithm rESPIRA
err(freq)       	:  8.1820e-14
err(weig)       	:  3.0103e-13
err([-10,10]^2)    	:  7.6275e-13
```
