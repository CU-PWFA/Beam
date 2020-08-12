#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:42:11 2018

@author: robert
"""

import numpy as np
from beam.elements import element


class Index(element.Element):
    """ A class representing a volume with a varying index of refraction.
    
    The class is meant to serve as the base class for any thick optical 
    elements.
    
    Parmaeters
    ----------
    Nx : int
        Number of grid points in the x direction, must match beam.
    Ny : int
        Number of grid points in the y direction, must match beam.
    Nz : int
        Number of grid points in the z direction
    X : int
        Width of the grid in the x direction, the grid goes from [-X/2, X/2).
        This must match the beam parameters.
    Y : int
        Width of the grid in the y direction, the grid goes from [-Y/2, Y/2).
        This must match the beam parameters.
    Z : int
        Length of the grid in the z direction, the grid goes from [0, Z].
    path : string
        The path for the calculation. This class will create a folder inside
        the path to store all output data in.
    name : string
        The name of the beam, used for naming files and folders.
    """
    keys = ['Nx',
            'Ny',
            'Nz',
            'X',
            'Y',
            'Z',
            'path',
            'name',
            'lam']
    
    # Initialization functions
    #--------------------------------------------------------------------------
    
    def __init__(self, params):
        super().__init__(params)
        self.k = 2*np.pi/self.lam
        self.create_grid()
        self.initialize_index()
        self.save_initial()
    
    def create_grid(self):
        """ Create an x-y-z rectangular grid. """
        X = self.X
        Y = self.Y
        Z = self.Z
        self.x = np.linspace(-X/2, X/2, self.Nx, False, dtype='double')
        self.y = np.linspace(-Y/2, Y/2, self.Ny, False, dtype='double')
        self.z = np.linspace(0, Z, self.Nz, False, dtype='double')
        
    def initialize_index(self, n=None):
        """ Create the array to store the index of refraction in. 
        
        Parameters
        ----------
        e : array-like, optional
            The array of field values to initialize the field to.
        """
        if n is None:
            self.n = np.zeros((self.Nx, self.Ny, self.Nz), dtype='double')
        else:
            self.n = n
        self.prep_index(self.n)
        self.save_index()
    
    def prep_index(self, n):
        """ Calculate the homogenous and inhomogenous parts from n. 
        
        Parmaeters
        ----------
        n : array-like
            Index of refraction (x, y, z) format.
        """
        self.nh = np.average(np.average(n, axis=0), axis=0)
        self.nih = n - self.nh
    
    def loadn(self, i):
        return self.nih[:, :, i]
    
    def loadnh(self, i):
        return self.nh[i]
    
    #File managment
    #--------------------------------------------------------------------------
    
    def save_initial(self):
        """ Save the initial params object and the grid. """
        super().save_initial()
        np.save(self.filePre + '_x.npy', self.x)
        np.save(self.filePre + '_y.npy', self.y)
        np.save(self.filePre + '_z.npy', self.z)
    
    def save_index(self):
        """ Save the phase mask to file. """
        #np.save(self.filePre + '_index.npy', self.n)