import numpy as np
import math
import cmath

def char_window_1d(n, s, xi):
    # generate one 1D characteristic window filter g_{s, \xi}(u) = 1_[0,s](u) \ast exp(i \xi u), u \in [0, n)
    # n: wavelet length
    # s: scale of the filter
    # xi: central frequency of filter
    
    # return: g_{s, \xi} and \hat(g)_{s, xi}, the filter and its Fourier transform
    x = np.arange(n)
    chi = np.zeros(n)
    chi[0:s] = 1
    o = np.exp(1j * xi * x)
    
    psi = np.multiply(chi, o)
    
    psi_hat = np.fft.fft(psi)
    return psi, psi_hat

def char_window_family_1d(n, s, xi):
    # generate a family of 1D characteristic window filters with specified scales and central frequencies in space
    # n: wavelet length
    # s: a sequence of scales
    # xi: a sequence of central frequencies
    
    # return: g (n * ns * nxi) and \hat(g), the filters and their Fourier transform
    ns = s.shape[0]
    nxi = xi.shape[0]
    
    psi = np.zeros((n, ns, nxi),dtype=complex)
    psi_hat = np.zeros((n, ns, nxi),dtype=complex)
    for i in range(ns):
        for k in range(nxi):
                psi[:, i, k], psi_hat[:, i, k] = char_window_1d(n, int(s[i]), xi[k])
    return psi, psi_hat

def scat_coeff_1d(x, g_hat):
    # compute 1st order scattering coefficients
    # Sx = {\int x(u) du, \int |x * \psi_{s, \xi}(u)| du} of nf + 1
    f = np.sum(np.abs(wave_trans_in_freq_1d(x, g_hat)), axis = 0)
    f = np.append(np.sum(x), f)
    return f

def wave_trans_in_freq_1d(x, psi_hat):
    # wavelet transform in frequency
    # x: signal of length n
    # psi_hat: filters of n * nf
    
    # return: wavelet transform n * nf
    
    x_hat = np.fft.fft(x)
    f = np.zeros(psi_hat.shape, dtype = complex)
    for i in range(psi_hat.shape[1]):
        f[:,i] = np.fft.ifft(np.multiply(x_hat, psi_hat[:,i]))
    return f

def determine_sigma(epsilon):
    sigma = np.sqrt(- 2 * np.log(epsilon)) / math.pi
    return sigma

def determine_J(N, Q, sigma, *alpha):
    if len(alpha) == 0:
        alpha = 3
    J = np.log2(N) - np.log2(alpha) - np.log2(sigma) - 1
    int_J = max(np.floor(J), 1);
    frac_J = (1/Q) * np.around((J - int_J) * Q);
    J = int_J + frac_J;
    return J