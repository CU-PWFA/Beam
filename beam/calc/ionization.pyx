#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: overflowcheck=False, cdivision=True
#cython: linetrace=False, binding=False
"""
Created on Mon Sep 25 16:45:42 2017

@author: robert
"""

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport abs

# Load necessary C functions
cdef extern from "complex.h" nogil:
    double complex cexp(double complex)
    double complex csqrt(double complex)

cdef extern from "math.h" nogil:
    double exp(double)
    double sqrt(double)
    double tgamma(double)
    double asinh(double)
    double atan(double)


cdef double adk_rate_static(double EI, double E, int Z, int l, int m, double lam) nogil:
    """ Calculates the ionization rate of a gas using the ADK model.

    Calculates the tunneling ionization rate of a gas in a constant electric
    field using the ADK model.

    Parameters
    ----------
    EI : double
        Ionization energy of the electron in eV.
    E : double
        Electric field strength in GV/m.
    Z : int
        Atomic residue i.e. which electron is being ionizaed (1st, 2nd...).
    l : int
        Orbital quantum number of the electron being ionized.
    m : int
        Magnetic quantum number of the electron being ionized.
    lam: double
        Wavelength of the laser in um.

    Returns
    -------
    w : double
        Ionization rate in 1/fs.
    """
    cdef double n = 3.68859*Z / sqrt(EI)
    cdef double E0 = EI**1.5
    # TODO replace the scipy gamma and factorial with a C version
    cdef double Cn2 = 4**n / (n*tgamma(2*n))
    cdef double N = 1.51927 * (2*l+1) * factorial(l+abs(m)) \
        / (2**abs(m) * factorial(abs(m)) * factorial(l-abs(m)))
    cdef double w = 0.0
    if E > 0:
        w = N * Cn2 * EI * (20.4927*E0/E)**(2*n-abs(m)-1) * exp(-6.83089*E0/E)
    return w


cdef double adk_rate_linear(double EI, double E, int Z, int l, int m, double lam) nogil:
    """ Calculates the ionization rate of a gas using the ADK model.

    Calculates the average tunneling ionization rate of a gas in a linearly
    polarized electric field. Use this function in conjunction with the
    envelope of the pulse to find the ionization fraction.

    Parameters
    ----------
    EI : double
        Ionization energy of the electron in eV.
    E : double
        Electric field strength in GV/m.
    Z : int
        Atomic residue i.e. which electron is being ionizaed (1st, 2nd...).
    l : int
        Orbital quantum number of the electron being ionized.
    m : int
        Magnetic quantum number of the electron being ionized.
    lam: double
        Wavelength of the laser in um.

    Returns
    -------
    w : double
        Ionization rate in 1/fs.
    """
    cdef double w = adk_rate_static(EI, E, Z, l, m, lam)
    w *= 0.305282*sqrt(E/EI**1.5)
    return w


cdef double ppt_rate_static(double EI, double E, int Z, int l, int m, double lam) nogil:
    """ Calculates the ionization rate of a gas using the PPT model.

    Calculates the tunneling ionization rate of a gas in a constant electric
    field using the PPT model.

    Parameters
    ----------
    EI : double
        Ionization energy of the electron in eV.
    E : double
        Electric field strength in GV/m.
    Z : int
        Atomic residue i.e. which electron is being ionizaed (1st, 2nd...).
    l : int
        Orbital quantum number of the electron being ionized.
    m : int
        Magnetic quantum number of the electron being ionized.
    lam: double
        Wavelength of the laser in um.

    Returns
    -------
    w : double
        Ionization rate in 1/fs.
    """
    cdef double alpha = 0.007297 # fine structure constant
    cdef double c = 2.9979e8 # speed of light, m/s
    cdef double m_e = 9.1094e-31 # electron mass, kg
    cdef double e = 1.6022e-19 # fundamental charge, C
    cdef double hbar = 1.0546e-34 # reduced planck constant, J*s
    cdef double eV = 1.6022e-19 # 1 electron volt, J
    
    cdef double n = alpha*c*sqrt(m_e/(2*e)) * Z/sqrt(EI)
    cdef double E0 = EI**1.5
    cdef double Cn2 = 4**n / (n*tgamma(2*n))
    cdef double N = (1e-15*e/hbar) * (2*l+1) * factorial(l+abs(m)) / (2**abs(m) * factorial(abs(m)) * factorial(l-abs(m)))
    cdef double k = ((2*3.14159*c/(lam*1e-6)) * sqrt(2*m_e) / e) * sqrt(EI*eV) / (E*1e9)
    cdef double g = (1.5/k) * ((1 + 1/(2*k**2))*asinh(k) - sqrt(1+k**2)/(2*k))
    # scipy fit of the Am parameter with m=0
    cdef double A0fit = 0.41087 * atan(-1.53960*k + 1.26086) + 0.64254
    cdef double A0 = 0.0
    if A0fit > 0:
        A0 = A0fit
    elif A0fit > 1:
        A0 = 1
    cdef double w = 0.0
    if E > 0:
        # 20.4927 = 2 * sqrt(8*m_e*e)/hbar / 1e9. Same as 2*E0/E with corrected E0 and E in V/m.
        # -6.83089 = -2/3 * sqrt(8*m_e*e)/hbar / 1e9. Same as -2/3*E0/E with corrected E0 and E in V/m.
        w = N * Cn2 * EI * A0 * (20.4927*E0/E)**(2*n-abs(m)-1) * exp(-6.83089*g*E0/E) * (1+k**2)**0.75
    return w


cdef double ppt_rate_linear(double EI, double E, int Z, int l, int m, double lam) nogil:
    """ Calculates the ionization rate of a gas using the PPT model.

    Calculates the average tunneling ionization rate of a gas in a linearly
    polarized electric field. Use this function in conjunction with the
    envelope of the pulse to find the ionization fraction.

    Parameters
    ----------
    EI : double
        Ionization energy of the electron in eV.
    E : double
        Electric field strength in GV/m.
    Z : int
        Atomic residue i.e. which electron is being ionizaed (1st, 2nd...).
    l : int
        Orbital quantum number of the electron being ionized.
    m : int
        Magnetic quantum number of the electron being ionized.
    lam: double
        Wavelength of the laser in um.

    Returns
    -------
    w : double
        Ionization rate in 1/fs.
    """
    cdef double m_e = 9.1094e-31 # kg
    cdef double e = 1.6022e-19 # C
    cdef double hbar = 1.0546e-34 # J*s
    cdef double w = ppt_rate_static(EI, E, Z, l, m, lam)
    cdef double E0 = EI**1.5 * sqrt(8*m_e*e)/hbar
    w *= sqrt((3*E*1e9) / (3.14159*E0))
    return w


cdef int factorial(int n) nogil:
    """ Calculate the factorial of an integer. """
    cdef int i
    cdef int ret = 1
    for i in range(n):
        ret *= i + 1
    return ret

cdef double rate_lithium(double EI, double E, int Z, int l, int m, double lam) nogil:
    """ Calculates the ionization rate of lithium using a TDSE model fit.
    
    Calculates the cycle averaged ionization rate using a simple exponential
    model fit to TDSE data and valid in the range of 3-9GV/m field. Only
    uses E for the calculation.
    
    Parameters
    ----------
    EI : double
        Ionization energy of the electron in eV.
    E : double
        Electric field strength in GV/m.
    Z : int
        Atomic residue i.e. which electron is being ionizaed (1st, 2nd...).
    l : int
        Orbital quantum number of the electron being ionized.
    m : int
        Magnetic quantum number of the electron being ionized.
    lam: double
        Wavelength of the laser in um.

    Returns
    -------
    w : double
        Ionization rate in 1/fs.
    """
    return 1.6e-6*E**6
