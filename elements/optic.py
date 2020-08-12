#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:33:15 2017

@author: robert
"""

import numpy as np
from beam.elements import element
import matplotlib.pyplot as plt


class Phase(element.Element):
    """ A class that represents some type of phase mask on a laser beam.
    
    This class is meant to serve as a base class for more complex phase masks
    such as lens, diffraction gratings, etc. Note, phase is in radians.
    
    Parameters
    ----------
    Nx : int
        Number of grid points in the x direction, must match beam.
    Ny : int
        Number of grid points in the y direction, must match beam.
    X : int
        Width of the grid in the x direction, the grid goes from [-X/2, X/2).
        This must match the beam parameters.
    Y : int
        Width of the grid in the y direction, the grid goes from [-Y/2, Y/2).
        This must match the beam parameters.
    path : string
        The path for the calculation. This class will create a folder inside
        the path to store all output data in.
    name : string
        The name of the beam, used for naming files and folders.
    """
    keys = ['Nx',
            'Ny',
            'X',
            'Y',
            'path',
            'name',
            'lam']
    
    # Initialization functions
    #--------------------------------------------------------------------------
    
    def __init__(self, params):
        super().__init__(params)
        self.k = 2*np.pi/self.lam
        if self.load is False:
            self.create_grid()
            self.initialize_phase()
            self.save_initial()        
    
    def create_grid(self):
        """ Create an x-y rectangular grid. """
        X = self.X
        Y = self.Y
        self.x = np.linspace(-X/2, X/2, self.Nx, False, dtype='double')
        self.y = np.linspace(-Y/2, Y/2, self.Ny, False, dtype='double')
        
    def initialize_phase(self, phi=None):
        """ Create the array to store the phase in. 
        
        Parameters
        ----------
        phi : array-like, optional
            The array of phase to initialize the mask.
        """
        if phi is None:
            self.phi = np.zeros((self.Nx, self.Ny), dtype='complex128')
        else:
            self.phi = phi
        self.save_phase()
        
    def load_element(self):
        """ Load the phase mask. """
        self.create_grid()
        self.load_phase()
    
    #File managment
    #--------------------------------------------------------------------------
    
    def save_initial(self):
        """ Save the initial params object and the grid. """
        super().save_initial()
        np.save(self.filePre + '_x.npy', self.x)
        np.save(self.filePre + '_y.npy', self.y)
    
    def save_phase(self):
        """ Save the phase mask to file. """
        np.save(self.filePre + '_phase.npy', self.phi)
        
    def load_phase(self):
        """ Load the phase of the mask. """
        self.phi = np.load(self.filePre + '_phase.npy')
        
        
class Intensity(element.Element):
    """ A class that represents some type of transmission mask on a laser beam.
    
    This class is meant to serve as a base class for more complex transmission masks
    such as such as apertures.
    
    Parameters
    ----------
    Nx : int
        Number of grid points in the x direction, must match beam.
    Ny : int
        Number of grid points in the y direction, must match beam.
    X : int
        Width of the grid in the x direction, the grid goes from [-X/2, X/2).
        This must match the beam parameters.
    Y : int
        Width of the grid in the y direction, the grid goes from [-Y/2, Y/2).
        This must match the beam parameters.
    path : string
        The path for the calculation. This class will create a folder inside
        the path to store all output data in.
    name : string
        The name of the beam, used for naming files and folders.
    """
    keys = ['Nx',
            'Ny',
            'X',
            'Y',
            'path',
            'name',
            'lam']
    
    # Initialization functions
    #--------------------------------------------------------------------------
    
    def __init__(self, params):
        super().__init__(params)
        self.create_grid()
        self.initialize_t()
        self.save_initial()
    
    def create_grid(self):
        """ Create an x-y rectangular grid. """
        X = self.X
        Y = self.Y
        self.x = np.linspace(-X/2, X/2, self.Nx, False, dtype='double')
        self.y = np.linspace(-Y/2, Y/2, self.Ny, False, dtype='double')
        
    def initialize_t(self, t=None):
        """ Create the array to store the mask in. 
        
        Parameters
        ----------
        t : array-like, optional
            The array of transmission fraction to initialize the mask.
        """
        if t is None:
            self.t = np.zeros((self.Nx, self.Ny), dtype='double')
        else:
            self.t = t
        self.save_t()
    
    #File managment
    #--------------------------------------------------------------------------
    
    def save_initial(self):
        """ Save the initial params object and the grid. """
        super().save_initial()
        np.save(self.filePre + '_x.npy', self.x)
        np.save(self.filePre + '_y.npy', self.y)
    
    def save_t(self):
        """ Save the transmission mask to file. """
        np.save(self.filePre + '_t.npy', self.t)


class SphericalLens(Phase):
    """ A phase mask that simulates a thin spherical lens.
    
    Parameters
    ----------
    f : double
        The focal length of the lens.
    """
    
    def __init__(self, params):
        self.keys.extend(
                ['f'])
        super().__init__(params)
    
    def initialize_phase(self):
        phi = -self.k * (self.x[:, None]**2+self.y[None, :]**2) / (2*self.f)
        super().initialize_phase(phi)
        

class AxiconLens(Phase):
    """ A phase mask that simulates a thin axicon lens.
    
    Parameters
    ----------
    beta : double
        The deflection angle of the axicon in deg.
    """
    
    def __init__(self, params):
        self.keys.extend(
                ['beta'])
        super().__init__(params)
    
    def initialize_phase(self):
        r = np.sqrt(self.x[:, None]**2+self.y[None, :]**2)
        phi = -self.k * np.radians(self.beta) * r
        super().initialize_phase(phi)


class Axilens(Phase):
    """ A phase mask that simulates a thin axilens.
    
    Parameters
    ----------
    R : double
        The radius of the input flattop.
    f0 : double
        The distance from the lens to the start of the focus.
    dz :
        The length of the focus.
    """
    
    def __init__(self, params):
        self.keys.extend(
                ['R',
                 'f0',
                 'dz'])
        super().__init__(params)
    
    def initialize_phase(self):
        r = np.sqrt(self.x[:, None]**2+self.y[None, :]**2)
        R = self.R
        f0 = self.f0
        dz = self.dz
        phi = -self.k*R**2 * np.log(f0+dz*r**2/R**2) / (2*dz)
        super().initialize_phase(phi)


class Aperture(Intensity):
    """ A transmission mask formed from a circular aperture.
    
    Parameters
    ----------
    r : double
        The radius of the aperture.
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['r'])
        super().__init__(params)
    
    def initialize_t(self):
        t = np.zeros((self.Nx, self.Ny), dtype='double')
        r2 = self.x[:, None]**2+self.y[None, :]**2
        sel = r2 < self.r**2
        t[sel] = 1.0
        super().initialize_t(t)


class Annulus(Intensity):
    """ A transmission mask formed from an annulus.

    Parameters
    ----------
    r_in : double
        The inner radius of the aperture.
    r_out : double
        The outer radius of th aperture.
    """
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['r_in',
                 'r_out'])
        super().__init__(params)

    def initialize_t(self):
        t = np.zeros((self.Nx, self.Ny), dtype='double')
        r2 = self.x[:, None]**2+self.y[None, :]**2
        sel = np.logical_and(r2 < self.r_out**2, r2 > self.r_in**2)
        t[sel] = 1.0
        super().initialize_t(t)
