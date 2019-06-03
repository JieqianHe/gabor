import numpy as np
import math
import cmath
pi = math.pi

def gabor_wavelet_space_2d(n,sigma,zeta,eta,theta,a,m):
    # generate one gabor wavelet with specified scale and rotation in space
    x = np.arange(-n/2,n/2)
    y = x
    X, Y = np.meshgrid(x,y)
    psi = np.zeros((n,n))

    X_new = a**(-m) * (X * np.cos(theta) + Y * np.sin(theta)); #rotate x and y
    Y_new = a**(-m) * (- X * np.sin(theta) + Y * np.cos(theta));
    # calculate gabor wavelet
    psi = a**(-2 * m) / (2 * math.pi * sigma**2 * zeta)* \
        np.exp((- X_new**2 - (Y_new/zeta)**2)/(2 * sigma**2)) * np.exp(1j * eta * X_new);
    return psi


def gabor_wavelet_family_space_2d(n,K,Q,S,sigma,zeta,eta,a):
    # generate a family of gabor wavelets with specified scales and rotations in space
    psi = np.zeros((n,n,K,Q*S),dtype=complex)
    
    for s in range(S):
        for q in range(Q):
            for k in range(K):
                psi[:, :, k, s*Q+q] = gabor_wavelet_space_2d(n,sigma,zeta,eta,k*math.pi/K,a,s+q/Q)
    return psi


def gabor_wavelet_freq_2d(n, sigma, zeta, eta, a,j,theta):
    # generate one gabor wavelet with specified scale and rotation in frequency
    pi = math.pi
    omega1 = np.linspace(-3*pi, 3*pi-(2*pi)/n, 3*n)
    omega2 = omega1

    omega1,omega2 = np.meshgrid(omega1,omega2)
    omega1_new = omega1 * np.cos(theta) + omega2 * np.sin(theta)
    omega2_new = - omega1 * np.sin(theta) + omega2 * np.cos(theta)

    psi_hat = np.exp(- 1/2 * sigma**2 * (a**j * omega1_new - eta)**2) * \
              np.exp(- 1/2 * sigma**2 * zeta**2 * (a**j * omega2_new)**2)

    psi_add = np.zeros((n,n))
    for i in range(3):
        for k in range(3):
            psi_add = psi_add + psi_hat[i * n :(i + 1) * n ,k * n:(k + 1) * n ]

    return psi_add


def gabor_wavelet_family_freq_2d(n,K,S,Q,sigma, zeta, eta,a):
    # generate a family of gabor wavelets with specified scales and rotations in frequency
    psi_hat = np.zeros((n,n,K,S*Q), dtype = complex)
    for s in range(S):
        for q in range(Q):
            for k in range(K):
                psi_hat[:,:,k,s*Q+q] = gabor_wavelet_freq_2d(n,sigma,zeta,eta,a,s+q/Q,k*math.pi/K)
    return psi_hat

def morlet_wavelet_space_2d(n, sigma, zeta, eta, theta, a, m):
    # generate one morlet wavelet and its 1st&2nd order derivatives with specified scale and rotation in space
    
    x = np.arange(-n/2,n/2)
    y = x
    X_temp, Y_temp = np.meshgrid(x,y)
    psi = np.zeros((n,n))
    ct = np.cos(theta)
    st = np.sin(theta)
    
    # define new axis by rotating and scaling x and y
    X = a**(-m) * (X_temp * ct + Y_temp * st)
    Y = a**(-m) * (- X_temp * st + Y_temp * ct)
    
    # define gaussian window
    g = 1 / (2 * math.pi * sigma**2 * zeta) * np.exp((- X**2 - (Y/zeta)**2)/(2 * sigma**2))
    # compute C
    C = np.sum(g * np.exp(1j * eta * X)) / np.sum(g) # numerically
#     C = np.exp( - sigma**2 * eta**2 / 2) # analytically

    # define mother morlet wavelet
    psi = a**(-2 * m) * g * (np.exp(1j * eta * X) - C)
    
    return psi

def morlet_wavelet_family_space_2d(n,K,Q,S,sigma,zeta,eta,a):
    # generate a family of morlet wavelets with specified scales and rotations in space
    # psi: [n, n,K, Q*S]
    # n: size of the wavelets should be n by n
    # K: number of different angles
    # Q*S: number of different scales
    
    psi = np.zeros((n,n,K,Q*S), dtype=complex) 
    
    for k in range(K):
        for s in range(S):
            for q in range(Q):
                psi[:, :,  k,s*Q+q] = morlet_wavelet_space_2d(n, sigma, zeta, eta, k*np.pi/K, a, s+q/Q)
    return psi

def morlet_wavelet_freq_2d(n, sigma, zeta, eta, theta, a, m):
    # generate one morlet wavelet and its 1st&2nd order derivatives with specified scale and rotation in frequency
    omega1 = np.arange(-n/2,n/2) * 2 * pi / n
    
    omega1 = np.linspace(-3*pi, 3*pi-(2*pi)/n, 3*n)
    omega2 = np.copy(omega1)
    omega1_temp, omega2_temp = np.meshgrid(omega1, omega2)
    
    ct = np.cos(theta)
    st = np.sin(theta)
    
    # define new axis by rotating and scaling x and y
    omega1 = a**m * (omega1_temp * ct + omega2_temp * st)
    omega2 = a**m * (- omega1_temp * st + omega2_temp * ct)
    
    # define morlet wavelet
    psi_hat = np.exp(- sigma**2 * (eta - omega1)**2 / 2 - sigma**2 * zeta **2 * omega2**2 / 2) - \
              np.exp(- sigma**2 * omega1**2 / 2 - sigma**2 * zeta**2 * omega2**2 / 2 - sigma**2 * eta**2 / 2)
        
    psi_add = np.zeros((n,n))
    for i in range(3):
        for k in range(3):
            psi_add = psi_add + psi_hat[i * n :(i + 1) * n ,k * n:(k + 1) * n ]

    return psi_add

def morlet_wavelet_family_freq_2d(n,K,Q,S,sigma,zeta,eta,a):
    # generate a family of morlet wavelets with specified scales and rotations in frequency
    # psi_hat: [n, n, K, Q*S]
    # n: size of the wavelets should be n by n
    # K: number of different angles
    # Q*S: number of different scales
    
    psi_hat = np.zeros((n,n,K,Q*S), dtype=complex) 
    
    for k in range(K):
        for s in range(S):
            for q in range(Q):
                psi_hat[:, :,  k,s*Q+q] = morlet_wavelet_freq_2d(n, sigma, zeta, eta, k*np.pi/K, a, s+q/Q)
    return psi_hat

def wave_trans_in_space_2d(x, psi):
    # wavelet transform in space
    npsi = psi.shape
    f = np.zeros(npsi,dtype = complex)
    for i in range(npsi[2]):
        for j in range(npsi[3]):
            f[:,:,i,j] = signal.convolve2d(x,psi[:,:,i,j],'same')
    return f

def wave_trans_in_freq_2d(x, psi_hat, p = False):
    # wavelet transform in frequency
    nx = x.shape
    if p:
        x_bar = pad2d(x, nx)
    else:
        x_bar = x
    x_hat = np.fft.fft2(x_bar)
    npsi = psi_hat.shape
    f = np.zeros(npsi, dtype = complex)
    for i in range(npsi[2]):
        for j in range(npsi[3]):
            f[:,:,i,j] = np.fft.ifft2(np.multiply(x_hat,np.fft.fftshift(psi_hat[:,:,i,j])))
    if p:
        f = f[nx[0]:2*nx[0], nx[1]:2*nx[1], :,:]
    return f

def pad2d(x, n):
    # pad 2d signal with size n in reflection
    s = x.shape
    
    w0 = min(s[0], n[0])
    a = np.flip(x[0:w0,:],axis = 0)
    b = np.flip(x[s[0] - w0:s[0], :], axis = 0)
    x_bar = np.concatenate((a,x,b), axis = 0)
    
    w1 = min(s[1], n[1])
    c = np.flip(x_bar[:,0:w1], axis = 1)
    d = np.flip(x_bar[:, s[1] - w1:s[1]], axis = 1)
    x_bar = np.concatenate((c,x_bar,d), axis = 1)
    
    return x_bar # (nx1 + 2*n) * (nx2 + 2*n)

# def wave_trans_in_freq_2d(x, psi_hat, p):
#     # wavelet transform in frequency
#     nx = x.shape
#     if p:
#         x_bar = pad2d(x, nx)
#     else:
#         x_bar = x
#     x_hat = np.fft.fft2(np.fft.fftshift(x_bar))
#     npsi = psi_hat.shape
#     f = np.zeros(npsi, dtype = complex)
#     for i in range(npsi[2]):
#         for j in range(npsi[3]):
#             f[:,:,i,j] = np.fft.fftshift(np.fft.ifft2(np.multiply(x_hat,np.fft.fftshift(psi_hat[:,:,i,j]))))
#     if p:
#         f = f[nx[0]:2*nx[0], nx[1]:2*nx[1], :,:]
#     return f

def wavelet_coefficients(x, psi_hat):
    f = wave_trans_in_freq_2d(x, psi_hat)
    mu = np.mean(np.abs(f), axis = (0,1))
    return mu

def scat_coeff_2d(x, psi_hat):
    # scattering coefficients of 1 dim: nf = 1 + nangle * nscales
    sx = wavelet_coefficients(x, psi_hat).flatten()
    sx = np.append(np.mean(x, axis = (0,1)), sx)
    return sx