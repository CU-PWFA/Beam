#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False
#cython: overflowcheck=False, cdivision=True
#cython: linetrace=False, binding=False
"""
Created on Mon Dec 11 10:58:22 2017

@author: robert
"""

import numpy as np
import warnings
cimport numpy as np
from numpy.fft import fftfreq
from scipy import integrate
from cython.parallel import prange

cdef extern from "math.h" nogil:
    double sin(double)
    double cos(double)
    double sqrt(double)

def electron_propagation_plasma(double[:, :] ptcls, double[:] z, double z0, 
                                double[:] ne, int dumpPeriod, saveP, dgammadz,
                                int n):
    """ Propagate an electron beam through an ion column.
    
    Propagates a collection of macro particles through a full blowout plasma
    wake. The calculation essentially assumes the electrons are propagating
    through a pure ion column and allows for energy increase/decrease.
    """
    cdef int i, j
    cdef int N = np.shape(ptcls)[0]
    cdef int Nz = len(z)
    cdef double dgamma
    cdef double kp, kb, dz
    cdef double coskb, sinkb, angle
    cdef double R11, R12, R21, R22
    cdef double x, y
    # Calculate parameters for each z-slice
    for i in range(Nz-1):
        kp = 5.95074e4 * sqrt(ne[i])
        if ne[i] < -1e-18:
            warnings.warn('Plasma density less than zero, treating as zero.')
        # Pre-calculate the energy gain per slice
        dz = z[i+1] - z[i]
        dgamma = 0.5 * dgammadz(ne[i]) * dz
        with nogil:
            for j in prange(N, num_threads=n):
        #if True:
        #    for j in range(N):
                ptcls[j, 5] += dgamma
                kb = kp/sqrt(2*ptcls[j, 5])
                #print('z: %0.2f, kb: %0.2f, kp: %0.2E, ne: %0.2E' %(z[i], kb, kp, ne[i]))
                coskb = cos(kb*dz)
                sinkb = sin(kb*dz)
                angle = 1 - 2*dgamma / ptcls[j, 5]
                # Calculate the components of the transfer matrix
                if ne[i] < 1.0e-18:
                    R11 = 1.0
                    R12 = dz
                    R21 = 0.0
                    R22 = 1.0
                else:
                    R11 = coskb
                    R12 = sinkb / kb
                    R21 = -angle * kb * sinkb
                    R22 = angle * coskb
                x = ptcls[j, 0]
                y = ptcls[j, 2]
                ptcls[j, 0] = R11 * x + R12 * ptcls[j, 1]
                ptcls[j, 1] = R21 * x + R22 * ptcls[j, 1]
                ptcls[j, 2] = R11 * y + R12 * ptcls[j, 3]
                ptcls[j, 3] = R21 * y + R22 * ptcls[j, 3]
                ptcls[j, 5] += dgamma
        if (i % dumpPeriod) == 0:
            saveP(ptcls, z[i+1]+z0)
    return np.array(ptcls)

def electron_propagation_nonlinear(double[:, :] ptcls, double[:] z, double z0, 
                                double[:] ne, int dumpPeriod, saveP, dgammadz,
                                int n, double a):
    """ Propagate an electron beam through an ion column with nonlinear focusing.
    
    Propagates a collection of macro particles through a full blowout plasma
    wake. The calculation includes a quadratic componenet to the focusing force.
    """
    cdef int i, j
    cdef int N = np.shape(ptcls)[0]
    cdef int Nz = len(z)
    cdef double dgamma
    cdef double kp, kb, dz
    cdef double coskb, sinkb, angle
    cdef double R11, R12, R21, R22
    cdef double x, y, r
    # Calculate parameters for each z-slice
    for i in range(Nz-1):
        kp = 5.95074e4 * sqrt(ne[i])
        if ne[i] < -1e-18:
            warnings.warn('Plasma density less than zero, treating as zero.')
        # Pre-calculate the energy gain per slice
        dz = z[i+1] - z[i]
        dgamma = 0.5 * dgammadz(ne[i]) * dz
        with nogil:
            for j in prange(N, num_threads=n):
        #if True:
        #    for j in range(N):
                ptcls[j, 5] += dgamma
                r = sqrt(ptcls[j, 0]**2 + ptcls[j, 2]**2)
                kb = kp/sqrt(2*ptcls[j, 5])*(1+a*r)
                #print('z: %0.2f, kb: %0.2f, kp: %0.2E, ne: %0.2E' %(z[i], kb, kp, ne[i]))
                coskb = cos(kb*dz)
                sinkb = sin(kb*dz)
                angle = 1 - 2*dgamma / ptcls[j, 5]
                # Calculate the components of the transfer matrix
                if ne[i] < 1.0e-18:
                    R11 = 1.0
                    R12 = dz
                    R21 = 0.0
                    R22 = 1.0
                else:
                    R11 = coskb
                    R12 = sinkb / kb
                    R21 = -angle * kb * sinkb
                    R22 = angle * coskb
                x = ptcls[j, 0]
                y = ptcls[j, 2]
                ptcls[j, 0] = R11 * x + R12 * ptcls[j, 1]
                ptcls[j, 1] = R21 * x + R22 * ptcls[j, 1]
                ptcls[j, 2] = R11 * y + R12 * ptcls[j, 3]
                ptcls[j, 3] = R21 * y + R22 * ptcls[j, 3]
                ptcls[j, 5] += dgamma
        if (i % dumpPeriod) == 0:
            saveP(ptcls, z[i+1]+z0)
    return np.array(ptcls)

def cs_propagation(double[:] z, double[:] ne, double beta0, double alpha0, 
                   double gb0, double dgdz0, double ne0, energy_model='witness'):
    """ Propagates the Courant_Snyder parameters through a plasma. 
    
    Calculates how the Courant-Snyder parameters evolve as the beam passes
    through a plasma.
    
    Prameters
    ---------
    z : array-like
        The grid in z, in meters.
    ne : array-like
        The plasma density on the grid in 10^17 cm^-3.
    beta0 : float
        The initial beta function at z[0], in meters.
    alpha0 : folat
        The inital value of the CS alpha at z[0].
    gb0 : float
        The inital relativisitc factor, gamma, of the beam.
    dgdz0 : function
        The change in the relativistic factor per unit length for a plasma ne0.
    ne0 : function
        The nominal plasma density dgdz0 is specified at.
    energy_model : string, optional
        Model to use for energy gain or loss in the plasma.
    
    Returns
    -------
    beta : array-like
        The beta function at each point in z, in meters.
    alpha : array-like
        The CS alpha at each point in z.
    gamma : array-like
        The CS gamma at each point in z, in meters^-1.
    gb : array-like
        The relativistic factor of the beam at each point in z.
    """
    cdef int i
    cdef int Nz = len(z)
    cdef double[:] beta = np.empty(Nz, dtype='double')
    cdef double[:] alpha = np.empty(Nz, dtype='double')
    cdef double[:] gamma = np.empty(Nz, dtype='double')
    cdef double[:] gb = np.empty(Nz, dtype='double')
    beta[0] = beta0
    alpha[0] = alpha0
    gamma[0] = (1+alpha0**2) / beta0
    gb[0] = gb0
    if energy_model == 'witness':
        dgammadz = dgammadz_witness
    if energy_model == 'drive':
        dgammadz = dgammadz_drive
    cdef double kp, dz, dgamma, kb, coskb, sinkb, cos2, sin2, cossin, ik   
    for i in range(Nz-1):
        kp = 5.95074e4 * sqrt(ne[i])
        if ne[i] < -1e-18:
            warnings.warn('Plasma density less than zero, treating as zero.')
        dz = z[i+1] - z[i]
        dgamma = 0.5 * dgammadz(ne[i], ne0, dgdz0) * dz
        # Begin calculating the values
        gb[i+1] = gb[i]+dgamma
        kb = kp/sqrt(2*gb[i+1])
        coskb = cos(kb*dz)
        sinkb = sin(kb*dz)
        cos2 = coskb**2
        sin2 = sinkb**2
        cossin = coskb*sinkb
        ik = 1/kb
        # propagate with the transfer matrix
        if ne[i] < 1.0e-18:
            beta[i+1]  = beta[i] - 2*dz*alpha[i] + dz**2*gamma[i]
            alpha[i+1] = alpha[i] - dz*gamma[i]
            gamma[i+1] = gamma[i]
        else:
            beta[i+1]  = cos2*beta[i] - 2*cossin*alpha[i]*ik + sin2*gamma[i]*ik**2
            alpha[i+1] = kb*cossin*beta[i] + (cos2-sin2)*alpha[i] - cossin*gamma[i]*ik
            gamma[i+1] = kb**2*sin2*beta[i] + 2*kb*cossin*alpha[i] + cos2*gamma[i]
        gb[i+1] += dgamma
    return beta, alpha, gamma, gb

cdef double dgammadz(double ne, double ne0, double dgdz0):
    cdef double eta = ne/ne0
    return dgdz0 * (2*eta - sqrt(eta))

cdef double dgammadz_witness(double ne, double ne0, double dgdz0):
    cdef double eta = ne/ne0
    return dgdz0 * eta**0.7985 * (2*eta - 1)

cdef double dgammadz_drive(double ne, double ne0, double dgdz0):
    cdef double eta = ne/ne0
    return dgdz0 * eta**0.71
