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
        self.create_grid()
        if self.load is False:
            self.save_initial()
        
    def create_grid(self):
        """ Create an x-y rectangular grid. """
        X = self.X
        Y = self.Y
        Z = self.Z
        self.x = np.linspace(-X/2, X/2, self.Nx, False, dtype='double')
        self.y = np.linspace(-Y/2, Y/2, self.Ny, False, dtype='double')
        self.z = np.linspace(0.0, Z, self.Nz, dtype='double')
        
    def initialize_plasma(self, n, ne):
        """ Save plasma and number densities, passed as an array, to file. """
        for i in range(self.Nz):
            self.save_num_density(n[:, :, i], i)
            self.save_plasma_density(ne[:, :, i], i)

    # Getters and setters
    #--------------------------------------------------------------------------
    
    def get_ne(self, z):
        """ Use interpolation to get the on axis plasma density on the grid z.
        
        Parameters
        ----------
        z : array-like
            Z grid to return the plasma density on.
        """
        Nz = self.Nz
        nePlasma = np.zeros(Nz, dtype='double')
        for i in range(Nz-1):
            if self.cyl is True:
                nePlasma[i] = self.load_plasma_density(i)[0][int(self.Nx/2)]
            else:
                nePlasma[i] = self.load_plasma_density(i)[0][int(self.Nx/2),
                        int(self.Ny/2)]
        if np.all(z == self.z):
            ne = nePlasma
        else: # We have to interpolate the plasma density
            tck = interpolate.splrep(self.z, nePlasma)
            ne = interpolate.splev(z, tck, ext=1) # Return 0 if out of range
        return ne
    
    #File managment
    #--------------------------------------------------------------------------
    
    def save_initial(self):
        """ Save the initial params object and the grid. """
        super().save_initial()
        np.save(self.filePre + '_x.npy', self.x)
        np.save(self.filePre + '_y.npy', self.y)
        np.save(self.filePre + '_z.npy', self.z)
    
    def save_plasma_density(self, ne, ind):
        """ Save the plasma density to file at the given z ind. """
        if self.cyl:
            ne = ne[:, int(self.Ny/2)]
        np.save(self.filePre + '_plasmaDensity_' + str(ind) + '.npy', ne)
    
    def save_num_density(self, n, ind):
        """ Save the total number density to file at the given z ind. """
        if self.cyl:
            n = n[:, int(self.Ny/2)]
        np.save(self.filePre + '_numberDensity_' + str(ind) + '.npy', n)
        
    def load_plasma_density(self, ind):
        """ Load the plasma density at the specified index. 
        
        Parameters
        ----------
        ind : int
            The save index to load the field at.
        
        Returns
        -------
        ne : array-like
            The plasma density at the specified index.
        z : double
            The z coordinate of the density.
        """
        ne = np.load(self.filePre + '_plasmaDensity_' + str(ind) + '.npy')
        z = self.z[ind]
        return ne, z
    
    def load_num_den(self, i):
        """ Returns a 2D array of the number density. """
        n = np.load(self.filePre + '_numberDensity_' + str(i) + '.npy')
        if self.cyl is True:
            x = self.x
            y = self.y
            n = self.reconstruct_from_cyl(x, n, x, y)
        return n
    
    def load_plasma_den(self, i):
        """ Returns a 2D array of plasma density. """
        ne = np.load(self.filePre + '_plasmaDensity_' + str(i) + '.npy')
        if self.cyl is True:
            x = self.x
            y = self.y
            ne = self.reconstruct_from_cyl(x, ne, x, y)
        return ne
    
    # Physics functions
    #--------------------------------------------------------------------------
    
    def get_plasma_number(self):
        """ Calculate the total number of plasma electrons ionized. """
        num = 0.0
        for i in range(len(self.z)):
            ne = self.load_plasma_den(i)
            num += np.sum(ne)
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        return num*dx*dy*dz*1e5
    
    def get_ionization_energy(self):
        """ Calculate the energy required just to ionize the plasma in joules. """
        return self.get_plasma_number()*self.atom['EI']*1.602e-19
        
    # Visualization functions
    #--------------------------------------------------------------------------
    
    def plot_long_density_center(self, lim=None):
        """ Plots the plasma density in an x-z plane at y=0. """
        Nz = self.Nz
        ne = np.zeros((Nz, self.Nx))
        if not self.cyl:
            for i in range(0, Nz-1):
                ne[i, :]= self.load_plasma_density(i)[0][:, int(self.Ny/2)]
        else:
            for i in range(0, Nz-1):
                ne[i, :] = self.load_plasma_density(i)[0]
        plt.figure(figsize=(7, 4), dpi=150)
        im = self.plot_long_density(ne, lim)
        plt.show(im)
    
    def plot_long_density(self, ne, lim=None):
        """ Create a longitudinal plasma density plot. """
        Z = self.Z
        X = self.X
        
        ne = self.prep_data(ne)
        im = plt.imshow(ne, aspect='auto', extent=[0, Z, -X/2, X/2])
        cb = plt.colorbar()
        cb.set_label(r'Plasma density ($\mathrm{10^{17}cm^{-3}}$)')
        plt.set_cmap('plasma')
        plt.xlabel(r'z')
        plt.ylabel(r'x')
        if lim is not None:
            plt.ylim(lim)
        plt.title('Longitudinal plasma density')
        return im
    
    def plot_profiles(self, H, dr, xlim=None, zlim=None):
        """ Plot longitudinal and transverse plasma density profiles. 
        
        Note, currently plots transverse lineouts in the middle third in Z. 
        
        Parameters
        ----------
        H : int
            The number of lineouts to draw.
        dr : double, optional
            Spacing between radial lineouts.
        """
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        X = self.X
        
        plt.figure(figsize=(16, 4.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        
        # Transverse density profiles
        ax1 = plt.subplot(gs[0, 0])
        sliceInd = np.linspace(Nz/3, 2*Nz/3, H, dtype=np.int)
        dz = self.Z/(Nz-1)
        slicePosition = (sliceInd-1)*dz/1e6
        slicePosition = ['%.2f' % num for num in slicePosition]
        ax1.set_prop_cycle('color',
                      [plt.cm.gist_rainbow(i) for i in np.linspace(0, 1, H)])
        for i in range(0, H):
            if not self.cyl:
                ne, z = self.load_plasma_density(sliceInd[i])[:, int(Ny/2)]
            else:
                ne, z = self.load_plasma_density(sliceInd[i])
            plt.plot(self.x, ne)
        plt.title('Transverse intensity profile at different distances')
        plt.xlabel('x ($\mu m$)')
        plt.ylabel('Plasma density')
        plt.legend(slicePosition, title='z in m')
        if xlim is not None:
            plt.xlim(xlim)
        plt.grid(True)
        
        # Longitudinal density profiles
        ax2 = plt.subplot(gs[0, 1])
        dx = X/(Nx-2)
        sliceInd = np.linspace(Nx/2, Nx/2+dr*H/dx, H, dtype=np.int)
        slicePosition = (sliceInd-1)*dx - X/2
        slicePosition = ['%.2f' % num for num in slicePosition]
        ax2.set_prop_cycle('color',
                      [plt.cm.winter(i) for i in np.linspace(0, 1, H)])
        for i in range(0, H):
            ne = np.zeros(Nz)
            if not self.cyl:
                for j in range(0, Nz-1):
                    ne[j] = self.load_plasma_density(j)[0][sliceInd[i], int(Ny/2)]
            else:
                for j in range(0, Nz-1):
                    ne[j] = self.load_plasma_density(j)[0][sliceInd[i]]
            plt.plot(self.z, ne)
        plt.title('Longitudinal intensity profiles for different radiuses')
        plt.xlabel('z (m)')
        plt.ylabel('Plasma density')
        plt.legend(slicePosition, title='Radius in  $\mu m$')
        if zlim is not None:
            plt.xlim(zlim)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class UniformPlasma(Plasma):
    """ A uniform density gas that can be ionized into a plasma. """
    
    def __init__(self, params):
        super().__init__(params)
    
    def load_num_den(self, i):
        """ Returns a 2D array of the number density, always constant. """
        return np.full((self.Nx, self.Ny), self.n0, dtype='double')
    
    def load_plasma_den(self, i):
        """ Returns a 2D array of plasma density, always 0. """
        return np.zeros((self.Nx, self.Ny), dtype='double')


class ExistingPlasma(Plasma):
    """ A uniform gas that is already partially ionized. 
    
    This class is meant to be used after a plasma has already been created and
    you would like to send an additional ionizing pulse through it. It loads
    the plasma density from the save files of a source plasma.
    """
    
    def __init__(self, params):
        self.keys.extend(
                ['sourcePath',
                 'sourceName'])
        super().__init__(params)
        self.sourceDir = self.sourcePath + 'elements/element_' \
                         + self.sourceName + '/'
        self.sourcePre = self.sourceDir + self.sourceName
        
    def load_num_den(self, i):
        """ Returns a 2D array of the number density, always constant. """
        return np.full((self.Nx, self.Ny), self.n0, dtype='double')
    
    def load_plasma_den(self, i):
        """ Returns a 2D array of plasma density. """
        ne = np.load(self.sourcePre + '_plasmaDensity_' + str(i) + '.npy')
        if self.cyl is True:
            x = self.x
            y = self.y
            ne = self.reconstruct_from_cyl(x, ne, x, y)
        return ne
