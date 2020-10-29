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


cdef double adk_rate_static(double EI, double E, int Z, int l, int m) nogil:
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


cdef double adk_rate_linear(double EI, double E, int Z, int l, int m) nogil:
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

    Returns
    -------
    w : double
        Ionization rate in 1/fs.
    """
    cdef double w = adk_rate_static(EI, E, Z, l, m)
    w *= 0.305282*sqrt(E/EI**1.5)
    return w


cdef int factorial(int n) nogil:
    """ Calculate the factorial of an integer. """
    cdef int i
    cdef int ret = 1
    for i in range(n):
        ret *= i + 1
    return ret

cdef double rate_lithium(double EI, double E, int Z, int l, int m) nogil:
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

    Returns
    -------
    w : double
        Ionization rate in 1/fs.
    """
    return 1.6e-6*E**6
