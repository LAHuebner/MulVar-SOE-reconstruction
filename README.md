# MulVar-SOE-reconstruction
This GitHub page is intended to store the entire code needed to recreate the results presented in the preprint [Parameter estimation for multivariate exponential sums viaiterative rational approximation](https://arxiv.org/abs/2504.19157). It is in joint work with Nadiia Derevianko.

The goal of this project is the recreation of multivariate SOE's (sums of exponentials) from their Fourier coefficients. To be more precisely, fix a SOE

$$f(t_1,\dots,t_d) = \sum_{j=1}^M \gamma_j e^{ t_1 \lambda_{j,1} + \dots+t_k \lambda_{j,d}}, $$

then its Fourier coefficients are given by

$$c_{\mathbf{k}}(f):= \frac{1}{P^d}\int_{[0,P]^d}f(\mathbf t) e^{-\frac{2\pi i}{P} \langle \mathbf k,\mathbf t\rangle}\mathrm d \mathbf{t}=\sum_{j=1}^M\gamma_j \prod_{\ell =1}^d \frac{e^{\lambda_{j,\ell}P}-1}{\lambda_{j,\ell}P-2\pi i k_\ell},\qquad k=(k_1,\dots,k_d) \in\mathbb Z^d.$$

The weights $\gamma_j$ and frequencies $\lambda_{j,k}$ are then reconstructed in terms of a rational reconstruction problem. For more details see the linked preprint.

# How to use the Code
Just download the files `mESPIRA.py` and `Examples.ipynb`, and store them in the same folder. Then all important routines can be imported from `mESPIRA.py` in the usual way, just type
```python
import 
```
