#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:08:43 2017

@author: robert
"""

import numpy as np
from beam.elements import element
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate

class Plasma(element.Element):
    """ A class that represents a gas/plasma that a laser can ionize. 
    
    This class stores the gas and plasma density for a region where a laser
    pulse can pass through. It contains information about the gas density and
    the state of ionization. Note that z denotes the start position of each
    cell, not the center. The plasma is only defined for cells [0:Nz-2], a
    total of Nz-1 cells.
    
    Parameters
    ----------
    Nx : int
        Number of grid points in the x direction, must match beam.
    Ny : int
        Number of grid points in the y direction, must match beam.
    Nz : int
        Number of grid points in the z direction.
    X : int
        Width of the grid in the x direction, the grid goes from [-X/2, X/2).
        This must match the beam parameters.
    Y : int
        Width of the grid in the y direction, the grid goes from [-Y/2, Y/2).
        This must match the beam parameters.
    Z : int
        Length of the grid in the z direction, the grid goes from[0, Z].
    n0 : double
        Nominal gas number density in 10^17 cm^-3.
    atom : dictionary
        EI : double
            Ionization energy in eV.
        Z : double
            Atomic residue i.e. which electron is being ionizaed (1st, 2nd...).
        l : double
            Orbital quantum number of the electron being ionized.
        m : double
            Magnetic quantum number of the electron being ionized.
        alpha : double
            Atomic polarizability of the gas in A^3.
    path : string
        The path for the calculation. This class will create a folder inside
        the path to store all output data in.
    name : string
        The name of the beam, used for naming files and folders.
    load : bool
        Boolean specifying if we are loading an existing object.
    cyl : bool
        Whether the plasma is cylindrically symmetric or not. Controls whether
        the entire transverse density is saved or only a 1D slice.
    """
    keys = ['Nx',
            'Ny',
            'Nz',
            'X',
            'Y',
            'Z',
            'n0',
            'atom',
            'path',
            'name',
            'load',
            'cyl']
    
    # Initialization functions
    #--------------------------------------------------------------------------
    
    def __init__(self, params):
        super().__init__(params)
        
    def initialize_plasma(self, n, ne):
        self.nez = ne
        self.nz = n
        self.z = np.linspace(0,self.Z,self.Nz)

    # Getters and setters
    #--------------------------------------------------------------------------
    
    def get_ne(self, z):
        """ Use interpolation to get the on axis plasma density on the grid z.
        
        Parameters
        ----------
        z : array-like
            Z grid to return the plasma density on.
        """
        nePlasma = self.nez
        if np.all(z == self.z):
            ne = nePlasma
        else: # We have to interpolate the plasma density
            tck = interpolate.splrep(self.z, nePlasma)
            ne = interpolate.splev(z, tck, ext=1) # Return 0 if out of range
        return ne
    
    def plot_long_density_center(self):
        """ Plots the plasma density in an x-z plane at y=0. """
        plt.figure(figsize=(10, 6))
        plt.plot(self.z,self.nez)
        plt.show()
