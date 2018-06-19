# -*- coding: utf-8 -*-
"""
Quantum Solve - Module for solving 1D and 2D waves

    This modules contains functions which use the Split-Step Fourier Method to 
    solve the Schrodinger equation and output either the wave function at each 
    time step or just the final wave function

@author: David Ellis 
"""

import numpy as np

def solve1D(psi0, x, V, dt, t_max):
    """
    Function for solving a 1D wave packet moving through potential V from t=0
    to t = t_max. Outputs only final wave function in x-space.
    
    Outputs
    ----------
    psix: array_like, complex
        length-N array of the final wave function calculated for time t = t_max
    
    Parameters
    ----------
    X: array_like, float
        length-N array of evenly spaced spatial coordinates
    psi0: array_like, complex
        length-N array of the initial wavefunction at time t = 0
    V: array_like, float
        length-N array giving the potential along x
    dt: float
        length of time step
    t_max: float
        maximum time for which solution should be calculated
    """
    
    # create array containing spacial frequencues
    N = len(x)    
    n = np.arange(-N/2,N/2,1)
    L = x[-1]-x[0]
    k = 2*n*np.pi/L
    
    # calculate number of loops required
    M = int(t_max/dt)

    psix = psi0    
    
    for n in range(M):
        # Advance half step in x-space
        psix_half = np.exp(-dt/2*1j*V)*psix 
        psik_half = np.fft.fftshift(np.fft.fft(psix_half)) 
        # Advance full step in k-space
        psik_full = np.exp(-dt/2*1j*k**2)*psik_half  
        psix_full = np.fft.ifft(np.fft.fftshift(psik_full)) 
        # Advance half step in x-space
        psix_full = np.exp(-dt/2*1j*V)*psix_full 
        psix = psix_full        
        
    return  psix
    
def solve1D_full(psi0, x, V, dt, t_max):
    """
    Function for solving a 1D wavepacket moving through potential V from t=0
    to t = t_max.  Outputs wave function for each time step in both 
    x-space and k-space.
    
    Outputs
    ----------
    solx: array_like, complex
        N by M array (M=int(t_max/dt)) containing the wave function in physical 
        (x) space for every time step from t = 0 to t = t_max      
    
    solk: array_like, complex
        N by M array (M=int(t_max/dt)) containing the wave function in Fourier 
        (k) space for every time step from t = 0 to t = t_max  
    
    Parameters
    ----------
    X: array_like, float
        length-N array of evenly spaced spatial coordinates
    psi0: array_like, complex
        length-N array of the initial wavefunction at time t = 0
    V: array_like, float
        length-N array giving the potential along x
    dt: float
        length of time step
    t_max: float
        maximum time for which solution should be calculated
    """
    
    # create array containing spacial frequencues
    N = len(x)    
    n = np.arange(-N/2,N/2,1)
    L = x[-1]-x[0]
    k = 2*n*np.pi/L
    
    # calculate number of loops required
    M = int(t_max/dt)
    
    solx = np.zeros((N, int(t_max/dt)), dtype = 'complex128')
    solk = np.zeros((N, int(t_max/dt)), dtype = 'complex128')
    
    psix = psi0    
    
    for n in range(M):
        # Advance half step in x-space
        psix_half = np.exp(-dt/2*1j*V)*psix 
        psik_half = np.fft.fftshift(np.fft.fft(psix_half)) 
        # Advance half step in k-space
        psik_full = np.exp(-dt/2*1j*k**2)*psik_half  
        psix_full = np.fft.ifft(np.fft.fftshift(psik_full)) 
        # Advance half step in x-space
        psix_full = np.exp(-dt/2*1j*V)*psix_full 
        psix = psix_full 
        solx[:,n] = psix_full
        solk[:,n] = psik_full
        psix = psix_full
    
    return solx, solk
    
def solve2D(psi0, X, Y, V, dt, t_max):
    """
    Function for solving a 2D wavepacket moving through potential V from t=0
    to t = t_max. Outputs only final wave function in physical (x-y) space.
    
    Outputs
    ----------
    psi: array_like, complex
        N by N array of the final wave function calculated for time t = t_max    
    
    Parameters
    ----------
    X: array_like, float
        N by N array of meshgrid x - coordinates
    Y: array_like, float
        N by N array of meshgrid y - coordinates
    psi0: array_like, complex
        N by N array of the initial wavefunction at time t = 0
    V: array_like, float
        N by N array of the the potential at each point in two dimensional 
        space
    dt: float
        length of time step
    t_max: float
        maximum time for which solution should be calculated
    """
    
    # create arrays containing 2D spacial frequencues
    N = len(X)    
    n = np.arange(-N/2,N/2,1)
    nv, = np.meshgrid(n)
    L = abs(X[0,-1]-X[0,1]) 
    k = 2*n*np.pi/L
    Kx, Ky = np.meshgrid(k, k)
    
    # calculate number of loops required
    M = int(t_max/dt)
    
    psi_full = psi0
    
    for n in range(M):
        # Advance half step in xy-space
        psi_half = np.exp(-dt/2*1j*(V*X**2 + V*Y**2))*psi_full 
        psik_half = np.fft.fftshift(np.fft.fft2(psi_half)) 
        # Advance half step in k-space
        psik_full = np.exp(-dt/2*1j*(Kx**2+Ky**2))*psik_half  
        psi_full = np.fft.ifft2(np.fft.fftshift(psik_full))
        # Advance half step in xy-space
        psi_full = np.exp(-dt/2*1j*V)*psi_full 

    return psi_full   
    
def solve2D_full(psi0, X, Y, V, dt, t_max):
    """
    Function for solving a 2D wavepacket moving through potential V from t=0
    to t = t_max. Outputs wave function for each time step in both 
    physical (x-y) space and k-space.
    
    Outputs
    ----------
    sol: array_like, complex
        N by N array (M=int(t_max/dt) of the wave function calculated for
        each time step from t = 0 to t = t_max in physical space
        
    solk: array_like, complex
        N by N array (M=int(t_max/dt) of the wave function calculated for
        each time step from t = 0 to t = t_max in Fourier (k) space  
    
    Parameters
    ----------
    X: array_like, float
        N by N array of meshgrid x - coordinates
    Y: array_like, float
        N by N array of meshgrid y - coordinates
    psi0: array_like, complex
        N by N array of the initial wavefunction at time t = 0
    V: array_like, float
        N by N array of the the potential at each point in two dimensional 
        space
    dt: float
        length of time step
    t_max: float
        maximum time for which solution should be calculated
    """
    
    # create array containing 2D spacial frequencues
    N = len(X)    
    n = np.arange(-N/2,N/2,1)
    nv, = np.meshgrid(n)
    L = abs(X[0,-1]-X[0,1])
    k = 2*n*np.pi/L
    Kx, Ky = np.meshgrid(k, k)
    
    # calculate number of loops required
    M = int(t_max/dt)
    
    sol = np.zeros((N, N, int(t_max/dt)), dtype = 'complex128')
    solk = np.zeros((N, N, int(t_max/dt)), dtype = 'complex128')
    
    psi_full = psi0
    
    for n in range(M):
        # Advance half step in xy-space
        psi_half = np.exp(-dt/2*1j*(V*X**2 + V*Y**2))*psi_full 
        psik_half = np.fft.fftshift(np.fft.fft2(psi_half)) 
        # Advance half step in k-space
        psik_full = np.exp(-dt/2*1j*(Kx**2+Ky**2))*psik_half  
        psi_full = np.fft.ifft2(np.fft.fftshift(psik_full)) 
        # Advance half step in xy-space
        psi_full = np.exp(-dt/2*1j*V)*psi_full 
        sol[:,:,n] = psi_full
        solk[:,:,n] = psik_full

    return sol
