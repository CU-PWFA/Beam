#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: overflowcheck=False, cdivision=True
#cython: linetrace=False, binding=False
"""
Created on Thu Sep 14 11:15:19 2017

@author: robert
"""

import numpy as np
cimport numpy as np
from numpy.fft import fftfreq
from scipy import integrate
from cython.parallel import prange

# Load necessary C functions
cdef extern from "complex.h" nogil:
    double complex exp(double complex)
    double complex sqrt(double complex)
    IF UNAME_SYSNAME != "Windows":
        double complex cexp(double complex)
        double complex csqrt(double complex)
    
IF UNAME_SYSNAME == "Windows":
    cdef double complex cexp(double complex x) nogil:
        return exp(x)
    cdef double complex csqrt(double complex x) nogil:
        return sqrt(x)


def fourier_prop(double complex[:, :] E, double[:] x, double[:] y, double[:] z,
                 double lam, double n, double z0, fft, ifft, save):
    """ Propagates an electromagnetic wave from a 2D boundary to an array of z.

    Uses the Rayleigh-Sommerfeld transfer function to propagate an
    electromagnetic wave from a 2D boundary. The calculation assumes a
    homogeneous index of refraction in the region of propagation. 

    Parameters
    ----------
    E : double complex[:, :]
        Array of E field values at points (x, y) along the boundary.
    x : double[:]
        Array of transverse locations in x on the boundary. Elements must be
        evenly spaced for fft.
    y : double[:]
        Array of transverse locations in y on the boundary. Elements must be
        evenly spaced for fft.
    z : double[:]
        Array of z distances from the boundary to calculate the field at. Does
        not need to be evenly spaced.
    lam : double
        Wavelength of the electromagnetic wave in vacuum.
    n : double, optional
        Index of refraction of the medium the wave is propagating through.
    z0 : double
        Position of the beam at the start of the function.
    fft : function
        The fft scheme, an object of the pyfftw.FFTW class.
    ifft : function
        The ifft scheme, an object of the pyfftw.FFTW class.
    save : function
        A function used to save the data, will be called with 
        'save(e, z[i], i)'. Will be called for each element of z.

    Returns
    -------
    e : double complex[:, :]
        The electric field at position (z, x, y).
    """
    cdef int i, j, k
    cdef int Nx = len(x)
    cdef int Ny = len(y)
    cdef int Nz = len(z)
    cdef double dx = x[1] - x[0]
    cdef double dy = y[1] - y[0]
    # Store the Fourier transform of the electric field on the boundary
    cdef double complex[:, :] e = np.zeros((Nx, Ny), dtype='complex128')
    cdef double complex[:, :] eb = np.zeros((Nx, Ny), dtype='complex128')
    eb = fft(E)
    # Pre-calculate the spatial frequencies
    cdef double[:] fx = fftfreq(Nx, dx)
    cdef double[:] fy = fftfreq(Ny, dy)
    cdef double complex[:, :] ikz = ikz_RS(fx, fy, lam, n)
    # Fourier transform, multiply by the transfer function, inverse Fourier
    for i in range(Nz):
        with nogil:
            for j in prange(Nx):
                for k in range(Ny):
                    e[j, k] = eb[j, k] * cexp(ikz[j, k]*z[i])
        e = ifft(e)
        save(e, z[i]+z0)
    return e


def pulse_prop(double complex[:, :, :] E, double[:] x, double[:] y, 
               double[:] z, double[:] t, double lam, double n, double z0, 
               fft, ifft, save):
    """
    Propagates an electromagnetic pulse from a 2D boundary to an array of z.

    Uses the Rayleigh-Sommerfeld transfer function to propagate slices of a
    pulse from a 2D boundary. The calculation assumes a
    homogeneous index of refraction in the region of propagation. 

    Parameters
    ----------
    E : double complex[:, :, :]
        Array of E field values at points (t, x, y) along the boundary.
    x : double[:]
        Array of transverse locations in x on the boundary. Elements must be
        evenly spaced for fft.
    y : double[:]
        Array of transverse locations in y on the boundary. Elements must be
        evenly spaced for fft.
    z : double[:]
        Array of z distances from the boundary to calculate the field at. Does
        not need to be evenly spaced.
    t : double[:]
        Array of t slices for the pulse. Does not need to be evenly spaced.
    lam : double
        Wavelength of the electromagnetic wave in vacuum.
    n : double, optional
        Index of refraction of the medium the wave is propagating through.
    z0 : double
        Position of the beam at the start of the function.
    fft : function
        The fft scheme, an object of the pyfftw.FFTW class.
    ifft : function
        The ifft scheme, an object of the pyfftw.FFTW class.
    save : function
        A function used to save the data, will be called with 
        'save(e, z[i], i)'. Will be called for each element of z.

    Returns
    -------
    e : double complex[:, :, :]
        The electric field at position (z, x, y).
    """
    cdef int i, j, k, l
    cdef int Nx = len(x)
    cdef int Ny = len(y)
    cdef int Nz = len(z)
    cdef int Nt = len(t)
    cdef double dx = x[1] - x[0]
    cdef double dy = y[1] - y[0]
    # Store the Fourier transform of the electric field on the boundary
    cdef double complex[:, :] e = np.zeros((Nx, Ny), dtype='complex128')
    cdef double complex[:, :, :] eb = np.zeros((Nt, Nx, Ny), dtype='complex128')
    for i in range(Nt):
        e = fft(E[i, :, :])
        eb[i, :, :] = e
    # Pre-calculate the spatial frequencies
    cdef double[:] fx = fftfreq(Nx, dx)
    cdef double[:] fy = fftfreq(Ny, dy)
    cdef double complex[:, :] ikz = ikz_RS(fx, fy, lam, n)
    # Fourier transform, multiply by the transfer function, inverse Fourier
    for i in range(Nz):
        for j in range(Nt):
            with nogil:
                for k in prange(Nx):
                    for l in range(Ny):
                        e[k, l] = eb[j, k, l] * cexp(ikz[k, l]*z[i])
            e = ifft(e)
            E[j, :, :] = e
        save(E, z[i]+z0)
    return E


def beam_prop(double complex[:, :] E, double[:] x, double[:] y, double[:] z,
              double lam, loadnh, double z0, fft, ifft, save, loadn):
    """ Propagates a em wave through a region with a non-uniform index. 
    
    Propagates an electromagnetic wave through a region with a non-uniform
    index of refraction. The index of refraction is expressed as a constant
    plus a perturbation (homogenous plus inhomogenous). 
    """
    cdef int i, j, k
    cdef int Nx = len(x)
    cdef int Ny = len(y)
    cdef int Nz = len(z)
    cdef double dx = x[1] - x[0]
    cdef double dy = y[1] - y[0]
    cdef double[:, :] nih = np.zeros((Nx, Ny), dtype='double')
    # Pre-calculate the spatial frequencies
    cdef double[:] fx = fftfreq(Nx, dx)
    cdef double[:] fy = fftfreq(Ny, dy)
    cdef double complex[:, :] ikz = ikz_RS(fx, fy, lam, loadnh(0))
    cdef double complex arg
    cdef double dz
    for i in range(1, Nz):
        ikz = ikz_RS(fx, fy, lam, loadnh(i))
        nih = (loadn(i) + loadn(i-1))/2;
        arg = 1j*2*np.pi*dz / lam
        E = fourier_step(E, ikz, dz, fft, ifft)
        with nogil:
            for j in prange(Nx):
                for k in range(Ny):
                    E[j, k] *= cexp(arg*nih[j, k])
        save(E, z[i]+z0)
        dz = z[i+1] - z[i]
    return E


cpdef double complex[:, :] fourier_step(double complex[:, :] E,
                   double complex[:, :] ikz, double dz, fft, ifft):
    """ Propagates a field across a single step of length dz.
    
    A lightweight version of fourier_prop meant to be integrated into other
    algorithms such as the split step method.
    
    Parameters
    ----------
    E : double complex[:, :]
        Array of initial E field values at points (x, y).
    ikz : double complex[:, :]
        Array of spatial frequencies in z, see ikz_RS.
    dz : double
        Size of the step to propagate.
    fft : function
        The fft scheme, an object of the pyfftw.FFTW class.
    ifft : function
        The ifft scheme, an object of the pyfftw.FFTW class.

    Returns
    -------
    e : double complex[:, :]
        The electric field after propagating a distance dz.
    """
    cdef int i, j
    cdef int Nx = np.shape(E)[0]
    cdef int Ny = np.shape(E)[1]
    cdef double complex[:, :] e = np.zeros((Nx, Ny), dtype='complex128')
    e = fft(E)
    # Fourier transform, multiply by the transfer function, inverse Fourier
    with nogil:
        for i in prange(Nx):
            for j in range(Ny):
                e[i, j] *= cexp(ikz[i, j]*dz)
    e = ifft(e)
    return e


cpdef double complex[:, :] ikz_RS(double[:] fx, double[:] fy, double lam,
                   double n):
    """ Calculates i*kz for the Rayleigh-Sommerfeld transfer function.
    
    Parameters
    ----------
    fx : double[:]
        The spatial frequencies in the x direction.
    fy : double[:]
        The spatial frequencies in the y direction.
    lam : double
        Wavelength of the electromagnetic wave in vacuum.
    n : double
        Index of refraction of the medium the wave is propagating through.

    Returns
    -------
    fz : double complex[:, :]
        The spatial frequency in the z direction.
    """
    cdef int i, j
    cdef int Nx = len(fx)
    cdef int Ny = len(fy)
    cdef double complex[:, :] ikz = np.zeros((Nx, Ny), dtype='complex128')
    cdef double[:] fx2 = np.zeros(Nx)
    cdef double[:] fy2 = np.zeros(Ny)
    cdef double complex pre = 1j*2*np.pi
    cdef double f2 = (n/lam)**2
    with nogil:
        for i in prange(Nx):
            fx2[i] = fx[i] * fx[i]
        for i in prange(Ny):
            fy2[i] = fy[i] * fy[i]
        for i in prange(Nx):
            for j in range(Ny):
                ikz[i, j] = pre * csqrt(f2 - fx2[i] - fy2[j])
    return ikz


cpdef double complex[:] fresnel_axis(double complex[:] E, double[:] r,
                   double[:] z, double lam, double n):
    """ Returns the electric field along the optical axis in the Fresnel limit.

    Uses the Fresnel diffraction integral to calculate the electric field along
    the optical axis resulting from a cylindrically symmetric felectric field.
    Not this is only valid in the paraxial limit. 

    Parameters
    ----------
    E : double complex[:]
        Array of E field values at points r on the boundary, E(r).
    r : double[:]
        Array of radial locations in on the boundary. Doesn't need to be evenly
        spaced, but should be ordered in increasing r.
    z : double[:]
        Array of z distances from the boundary to calculate the field at. Does
        not need to be evenly spaced.
    lam : double
        Wavelength of the electromagnetic wave in vacuum.
    n : double
        Index of refraction of the medium the wave is propagating through.

    Returns
    -------
    e : double complex[:]
        The electric field at position (z).
    """
    cdef int i
    cdef int Nr = len(r)
    cdef int Nz = len(z)
    cdef double k = 2*np.pi*n/lam
    cdef double complex pre
    cdef double complex[:] e = np.zeros(Nz, dtype='complex128')
    cdef double complex[:] arg = np.zeros(Nr, dtype='complex128')
    # TODO make this parallel by writing/finding a cython integrator
    for i in range(Nz):
        if z[i] == 0.0: # Throw an error, this isn't paraxial
            raise ValueError('z must not equal 0, abs(z) should be several \
                             times larger than the maximum value of r')
        else:
            pre = k * cexp(1j*k*z[i]) / (1j*z[i])
            for j in range(Nr):
                arg[j] = E[j] * cexp(1j*k*r[j]*r[j]/(2*z[i])) * r[j]
            e[i] = pre * integrate.simps(arg, r)
    return e
            
