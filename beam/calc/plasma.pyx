#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: overflowcheck=False, cdivision=True
#cython: linetrace=False, binding=True
"""
Created on Mon Sep 18 16:59:45 2017

@author: robert
"""

import numpy as np
cimport numpy as np
from numpy.fft import fftfreq
from scipy import integrate
from cython.parallel import prange
from beam.calc import laser
from beam.calc.ionization cimport adk_rate_linear
from beam.calc.ionization cimport ppt_rate_linear
from beam.calc.ionization cimport rate_lithium

# Load necessary C functions
cdef extern from "complex.h" nogil:
    # double complex cexp(double complex)
    double complex exp(double complex)
    double complex csqrt(double complex)
    double cabs(double complex)

cdef extern from "math.h" nogil:
    double exp(double)
    double sqrt(double)


def plasma_refraction(double complex[:, :, :] E, double[:] x, double[:] y,
                      double[:] z, double[:] t, double lam, double n0, 
                      double z0, fft, ifft, saveE, saven, atom, loadn, loadne,
                      int num_threads, ionization='adk'):
    """ Propagate a laser pulse through a plasma accounting for refraction.

    Propogates a laser pulse through a region of partially ionized gas. This
    function accounts for refraction from the plasma. It determines the plasma
    density by calculating the ionization that has resulted from each temporal
    piece of the pulse. The results are stored in a file, only the central x-z
    plane is recorded. Note that the z array is effectively shifted so z[0]=0.
    
    Parameters
    ----------
    E : array of complex
        Electric field in t, x, and y.
    x : array of double
        X grid.
    y : array of double
        Y grid.
    z : array of double
        The z array to propagate through, step sizes can be variable.
    t : array of double
        T grid.
    lam : double
        Wavelength of the laser in um.
    n0 : double
        Background gas density for the diffraction step.
    z0 : double
        Initial z value to correctly set z in the save files.
    fft : fft object
        Pyfftw fft object.
    ifft : ifft object
        Pyfftw ifft object.
    saveE : func
        The function should take two arguments, the field and the z position.
    saven : func
        The function should take two arguments, a transverse plasma density slice,
        and an index.
    atom : ionization.atom object
        The object with the ionization parameters for the object.
    loadn : func
        Function that loads the initial gas density in 10^17 cm^-3 when passed an index.
    loadne : func
        Function that loads the initial plasma density in 10^17 cm^-3 when passed an index.
    ionization : string, optional
        Function to use for the ionization model.
    """
    cdef int i, j, k, l
    # TODO abstract this into its own function
    cdef int Nx = len(x)
    cdef int Ny = len(y)
    cdef int Nz = len(z)
    cdef int Nt = len(t)
    cdef double dx = x[1] - x[0]
    cdef double dy = y[1] - y[0]
    cdef double dt = t[1] - t[0]
    # Get the ionization information from atom
    cdef double EI = atom['EI']
    cdef int Z = atom['Z']
    cdef int ll = atom['l']
    cdef int m = atom['m']
    # Plasma density and index of refraction arrays
    # n is total number density, ng + ne
    cdef double[:, :] n = np.zeros((Nx, Ny), dtype='double')
    cdef double[:, :] ne = np.zeros((Nx, Ny), dtype='double')
    cdef double ngas = atom['alpha'] *  6.283e-07
    cdef double nplasma = plasma_index(1.0, lam) - 1.0
    cdef double nh = 1.0 + ngas*n0
    cdef double dn = nplasma - ngas
    # Pre-calculate the spatial frequencies
    cdef double[:] fx = fftfreq(Nx, dx)
    cdef double[:] fy = fftfreq(Ny, dy)
    cdef double complex[:, :] ikz = laser.ikz_RS(fx, fy, lam, nh)
    if ionization == 'adk':
        rate_func = adk_rate_linear
    if ionization == 'ppt':
        rate_func = ppt_rate_linear
    if ionization =='lithium':
        rate_func = rate_lithium
    cdef double complex arg
    cdef double rate
    cdef double Eavg
    cdef double dz
    cdef double complex[:, :] e = np.zeros((Nx, Ny), dtype='complex128')
    for i in range(1, Nz):
        n = loadn(i-1)
        ne = loadne(i-1)
        dz = z[i] - z[i-1]
        arg = 1j*2*np.pi*dz*dn / lam
        for j in range(Nt):
            # Propagate the beam through
            e = laser.fourier_step(E[j, :, :], ikz, dz, fft, ifft)
            with nogil:
                for k in prange(Nx, num_threads=num_threads):
                    for l in range(Ny):
                        # e[k, l] *= cexp(arg*ne[k, l])
                        e[k, l] *= exp(arg*ne[k, l])
                        # Ionize the gas
                        Eavg = 0.5*(cabs(E[j, k, l]) + cabs(e[k, l]))
                        rate = rate_func(EI, Eavg, Z, ll, m, lam)
                        ne[k, l] = n[k, l]-(n[k, l]-ne[k, l])*exp(-rate*dt)
            E[j, :, :] = e
        saveE(E, z[i]+z0)
        saven(ne, i-1)
        for k in range(Nx):
            for l in range(Ny):
                  ne[k, l] = 0.0                           
    return E


def plasma_refraction_energy(double complex[:, :, :] E, double[:] x, double[:] y,
                      double[:] z, double[:] t, double lam, double n0, 
                      double z0, fft, ifft, saveE, saven, atom, 
                      loadn, loadne, int num_threads, double temp=0.0,
                      double n2=0.0, ionization='adk'):
    """ Propagate a laser pulse through a plasma accounting for refraction.

    Propogates a laser pulse through a region of partially ionized gas. This
    function accounts for refraction from the plasma. It determines the plasma
    density by calculating the ionization that has resulted from each temporal
    piece of the pulse. The results are stored in a file, only the central x-z
    plane is recorded. Note that the z array is effectively shifted so z[0]=0.
    Energy is also taken out of the pulse for ionization and heating. The energy
    is removed in the cell the energy.
    
    Parameters
    ----------
    E : array of complex
        Electric field in t, x, and y.
    x : array of double
        X grid.
    y : array of double
        Y grid.
    z : array of double
        The z array to propagate through, step sizes can be variable.
    t : array of double
        T grid.
    lam : double
        Wavelength of the laser in um.
    n0 : double
        Background gas density for the diffraction step.
    z0 : double
        Initial z value to correctly set z in the save files.
    fft : fft object
        Pyfftw fft object.
    ifft : ifft object
        Pyfftw ifft object.
    saveE : func
        The function should take two arguments, the field and the z position.
    saven : func
        The function should take two arguments, a transverse plasma density slice,
        and an index.
    atom : ionization.atom object
        The object with the ionization parameters for the object.
    loadn : func
        Function that loads the initial gas density in 10^17 cm^-3 when passed an index.
    loadne : func
        Function that loads the initial plasma density in 10^17 cm^-3 when passed an index.
    temp : double, optional
        The final temperature of the plasma in eV for energy loss. Currently taken
        out of the field locally.
    n2 : double, optional
        The nonlinear index of refraction at atmospheric pressure. In cm^2/W.
    ionization : string, optional
        Function to use for the ionization model.
    """
    cdef int i, j, k, l
    # TODO abstract this into its own function
    cdef int Nx = len(x)
    cdef int Ny = len(y)
    cdef int Nz = len(z)
    cdef int Nt = len(t)
    cdef double dx = x[1] - x[0]
    cdef double dy = y[1] - y[0]
    cdef double dt = t[1] - t[0]
    # Get the ionization information from atom
    cdef double EI = atom['EI']
    cdef int Z = atom['Z']
    cdef int ll = atom['l']
    cdef int m = atom['m']
    # Plasma density and index of refraction arrays
    # n is total number density, ng + ne
    # Note, this assumes the plasma density is well below the critical density
    cdef double[:, :] n = np.zeros((Nx, Ny), dtype='double')
    cdef double[:, :] ne = np.zeros((Nx, Ny), dtype='double')
    cdef double ngas = atom['alpha'] * 6.283e-07
    cdef double nplasma = plasma_index(1.0, lam) - 1.0
    cdef double nh = 1.0 + ngas*n0
    cdef double dn = nplasma - ngas
    # n2 is measured at atmospheric pressure, calculate it per 1e17cm^-3
    cdef double dn2 = n2 * 3.99361e-3
    dn2 = dn2 * 1.32721e11 # Convert so we can multiply be GeV^2 later
    # Pre-calculate the spatial frequencies
    cdef double[:] fx = fftfreq(Nx, dx)
    cdef double[:] fy = fftfreq(Ny, dy)
    cdef double complex[:, :] ikz = laser.ikz_RS(fx, fy, lam, nh)
    if ionization == 'adk':
        rate_func = adk_rate_linear
    if ionization == 'ppt':
        rate_func = ppt_rate_linear
    if ionization =='lithium':
        rate_func = rate_lithium
    cdef double complex arg
    cdef double rate
    cdef double Eavg
    cdef double dz
    cdef double ne_new
    cdef double dE
    cdef double ng
    cdef double e_abs
    cdef double complex arg_kerr
    cdef double complex[:, :] e = np.zeros((Nx, Ny), dtype='complex128')
    for i in range(1, Nz):
        n = loadn(i-1)
        ne = loadne(i-1)
        dz = z[i] - z[i-1]
        arg = 1j*2*np.pi*dz*dn / lam
        arg_kerr = 1j*2*np.pi*dz*dn2 / lam 
        for j in range(Nt):
            # Propagate the beam through
            e = laser.fourier_step(E[j, :, :], ikz, dz, fft, ifft)
            with nogil:
                for k in prange(Nx, num_threads=num_threads):
                    for l in range(Ny):
                        ng = n[k, l] - ne[k, l]
                        e_abs = cabs(e[k, l])
                        # e[k, l] *= cexp(arg*ne[k, l] + arg_kerr*ng*e_abs*e_abs)
                        e[k, l] *= exp(arg*ne[k, l] + arg_kerr*ng*e_abs*e_abs)
                        # Ionize the gas
                        Eavg = 0.5*(cabs(E[j, k, l]) + e_abs)
                        rate = rate_func(EI, Eavg, Z, ll, m, lam)
                        ne_new = n[k, l]-ng*exp(-rate*dt)
                        # Remove energy from the laser
                        dE = energy_loss(ne[k, l], ne_new, EI+temp, dz, dt, e[k, l])
                        ne[k, l] = ne_new
                        e[k, l] *= dE
            E[j, :, :] = e
        saveE(E, z[i]+z0)
        saven(ne, i-1)
    return E

# TODO, move this to a helper function file
cpdef double plasma_index(double n, double lam):
    """ Calculates the index of refraction of a plasma.

    Parameters
    ----------
    n : double
        Density of the plasma in 10^17 cm^-3.
    lam : double
        Wavelength of the incident light in um.
    """
    return 1.0 - n * lam*lam * 4.47869e-5

cdef double energy_loss(double n_i, double n_f, double EI, double dz, double dt, double complex E0) nogil:
    """ Calculate the decrease in field strength from ionization energy loss.
    
    Parameters
    ----------
    n_i : double
        Plasma density at the beginning of the step in 10^17 cm^-3.
    n_f : double
        Plasma density at the end of the step in 10^17 cm^-3.
    EI : double
        Ionization energy of the atom in eV.
    dz : double
        Length of the cell in um.
    dt : double
        Length of the time slice in fs.
    E0 : double complex
        Current electric field of the pulse.
    """
    cdef double a
    cdef double absE
    absE = cabs(E0)
    a = 1.207e-2*EI*(n_f-n_i)*dz / (dt*absE*absE)
    if a < 1:
        return sqrt(1-a)
    else:
        return 0.0
    
