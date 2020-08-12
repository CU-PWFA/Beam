#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:47:12 2017

@author: robert
"""

import pyfftw
import numpy as np
from beam.beams import beam
from beam.calc import laser
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fftshift


class Laser(beam.Beam):
    """ A laser beam class that stores the field on a two dimensional grid. 
    
    This class stores a two dimensional grid upon which a complex scalar field
    is stored. The grid has a point at exactly 0, (Nx/2, Ny/2). The grid is
    meant for use with the discrete fourier transform. A note on units, any
    unit of length may be used as long as it is consistent for all variables.
    Generally, units of microns are appropriate.
    
    Parameters
    ----------
    Nx : int
        Number of grid points in the x direction, a power of 2 is recommended.
    Ny : int
        Number of grid points in the y direction, a power of 2 is recommended.
    X : double
        Width of the grid in the x direction, the grid goes from [-X/2, X/2).
    Y : double
        Width of the grid in the y direction, the grid goes from [-Y/2, Y/2).
    lam : double
        The vacuum wavelength of the laser radiation.
    path : string
        The path for the calculation. This class will create a folder inside
        the path to store all output data in.
    name : string
        The name of the beam, used for naming files and folders.
    load : bool
        Boolean specifying if we are loading an existing object.
    threads : int
        The number of processors to parallelize the fft over.
    cyl : bool
        Whether the beam is cylindrically symmetric or not. Controls whether
        the entire transverse field is saved or only a 1D slice.
    """
    keys = ['Nx',
            'Ny',
            'X',
            'Y',
            'lam',
            'path',
            'name',
            'load',
            'threads',
            'cyl']
    
    # Initialization functions
    #--------------------------------------------------------------------------
    
    def __init__(self, params):
        super().__init__(params)
        self.k = 2*np.pi / self.params['lam']
        # Create internal variables
        self.create_fft()
        if self.load is False:
            self.create_grid()
            self.initialize_field()
            self.save_initial()        
    
    def create_grid(self):
        """ Create an x-y rectangular grid. """
        X = self.X
        Y = self.Y
        self.x = np.linspace(-X/2, X/2, self.Nx, False, dtype='double')
        self.y = np.linspace(-Y/2, Y/2, self.Ny, False, dtype='double')
        self.z = []
    
    def create_fft(self):
        """ Create the fftw plans. """
        threads = self.threads
        # Allocate space to carry out the fft's in
        efft = pyfftw.empty_aligned((self.Nx, self.Ny), dtype='complex128')
        self.fft = pyfftw.builders.fft2(efft, overwrite_input=True,
                                         avoid_copy=True, threads=threads)
        self.ifft = pyfftw.builders.ifft2(efft, overwrite_input=True, 
                                           avoid_copy=True, threads=threads)
        
    def initialize_field(self, e=None):
        """ Create the array to store the electric field values in. 
        
        Parameters
        ----------
        e : array-like, optional
            The array of field values to initialize the field to.
        """
        if e is None:
            self.e = np.zeros((self.Nx, self.Ny), dtype='complex128')
        else:
            self.e = np.array(e, dtype='complex128')
        self.saveInd = 0
        self.z = []
        self.save_field(self.e, 0.0)
    
    def load_beam(self):
        """ Load the beam, specifically load the z grid and saveInd. """
        self.create_grid()
        self.z = np.load(self.filePre + '_z.npy')
        self.saveInd = len(self.z)
        e, z = self.load_field(self.saveInd - 1)
        if not self.cyl:
            self.e = e
        else:
            x = self.x
            y = self.y
            self.e = self.reconstruct_from_cyl(x, e, x, y)
        
    # Getters and setters
    #--------------------------------------------------------------------------
        
    def set_field(self, e):
        """ Set the value of the electric field. """
        self.e = np.array(e, dtype='complex128')
        self.save_field(self.e, self.z[-1])
        
    def get_dx(self):
        """ Get the grid spacing. """
        x = self.x
        return x[1] - x[0]
    
    def get_dy(self):
        """ Get the grid spacing. """
        y = self.y
        return y[1] - y[0]
        
    def get_f(self):
        """ Get the spatial frequencies of the fft of e. """
        dx = self.get_dx()
        dy = self.get_dy()
        fx = fftfreq(self.Nx, dx)
        fy = fftfreq(self.Ny, dy)
        return fx, fy
        
    # File managment
    #--------------------------------------------------------------------------
        
    def save_initial(self):
        """ Save the initial params object and the grid. """
        super().save_initial()
        np.save(self.filePre + '_x.npy', self.x)
        np.save(self.filePre + '_y.npy', self.y)
    
    def save_field(self, e, z):
        """ Save the current electric field to file and adavnce z. """
        if self.cyl:
            e = e[:, int(self.Ny/2)]
        np.save(self.filePre + '_field_' + str(self.saveInd) + '.npy', e)
        self.saveInd += 1
        self.z.append(z)
        self.save_z()
        
    def save_z(self):
        """ Save the z array. """
        np.save(self.filePre + '_z.npy', self.z)
        
    def load_field(self, ind):
        """ load the electric field at the specified index. 
        
        Parameters
        ----------
        ind : int
            The save index to load the field at.
        
        Returns
        -------
        e : array-like
            The electric field at the specified index.
        z : double
            The z coordinate of the field.
        """
        e = np.load(self.filePre + '_field_' + str(ind) + '.npy')
        z = self.z[ind]
        return e, z
        
    # Physics functions
    #--------------------------------------------------------------------------
        
    def propagate(self, z, n):
        """ Propagate the field to an array of z distances.
        
        Prameters
        ---------
        z : array-like
            Array of z distances from the current z to calculate the field at. 
            Does not need to be evenly spaced.
        n : double
            Index of refraction of the medium the wave is propagating through.
        """
        z = np.array(z, ndmin=1, dtype='double')
        self.e = laser.fourier_prop(self.e, self.x, self.y, z, self.lam, n, 
                                    self.z[-1], self.fft, self.ifft,
                                    self.save_field)
        self.e = np.array(self.e, dtype='complex128')
    
    # Visualization functions
    #--------------------------------------------------------------------------
    
    def plot_current_intensity(self, lim=None):
        """ Plots the current intensity of the beam. """
        im = self.plot_intensity(self.e, self.z[-1], lim)
        plt.show(im)
        
    def plot_intensity_at(self, ind, lim=None):
        """ Plots the intensity at a particular z distance.
        
        Parameters
        ----------
        ind : int
            The save index to plot the field at, see the _z file to find z.
        """
        e, z = self.load_field(ind)
        if not self.cyl:
            data = e
        else:
            x = self.x
            y = self.y
            data = self.reconstruct_from_cyl(x, e, x, y)
        im = self.plot_intensity(data, z, lim)
        plt.show(im)
    
    def plot_intensity(self, e, z, lim=None):
        """ Create an intensity plot. """
        X = self.X
        Y = self.Y
        
        I = self.intensity_from_field(e)
        I = self.prep_data(I)
        im = plt.imshow(I, aspect='auto', extent=[-X/2, X/2, -Y/2, Y/2])
        cb = plt.colorbar()
        cb.set_label(r'Intensity')
        plt.set_cmap('viridis')
        plt.xlabel(r'x')
        plt.ylabel(r'y')
        if lim is not None:
            plt.xlim(lim)
            plt.ylim(lim)
        plt.title('Transverse intensity at z=%.2f' % z)
        return im
    
    def plot_current_field(self, xlim=None, flim=None, log=False, wrap_order=0):
        beam = self
        I = beam.intensity_from_field(beam.e)
        If = abs(fftshift(beam.fft(beam.e)))**2
        fx, fy = beam.get_f()
        fx = fftshift(fx)
        fy = fftshift(fy)
        phase = np.angle(beam.e)

        # Images
        X = beam.X
        Y = beam.Y
        ext = [-X/2, X/2, -Y/2, Y/2]
        extf = [fx[0], fx[-1], fy[0], fy[-1]]
        plt.figure(figsize=(16, 4), dpi=150)
        plt.subplot(131)
        plt.imshow(beam.prep_data(I), aspect='auto', extent=ext, cmap='viridis')
        cb = plt.colorbar()
        cb.set_label(r'Intensity ($10^{14}$ W/cm^2)')
        plt.xlabel(r'$x$ (um)')
        plt.ylabel(r'$y$ (um)')
        if xlim != None:
            plt.xlim(xlim)
            plt.ylim(xlim)
        
        if wrap_order == 0:
            axis0 = 0
            axis1 = 1
        elif wrap_order == 1:
            axis0 = 1
            axis1 = 0
        plt.subplot(132)
        plt.imshow(np.unwrap(np.unwrap(beam.prep_data(phase), axis=axis0), axis=axis1), aspect='auto', extent=ext, cmap='viridis')
        cb = plt.colorbar()
        cb.set_label(r'Phase (rad)')
        plt.xlabel(r'$x$ (um)')
        plt.ylabel(r'$y$ (um)')
        if xlim != None:
            plt.xlim(xlim)
            plt.ylim(xlim)

        plt.subplot(133)
        plt.imshow(beam.prep_data(If), aspect='auto', extent=extf, cmap='viridis')
        cb = plt.colorbar()
        cb.set_label(r'Intensity (arb unit)')
        plt.xlabel(r'$f_x$ (um$^{-1}$)')
        plt.ylabel(r'$f_y$ (um$^{-1}$)')
        if flim != None:
            plt.xlim(flim)
            plt.ylim(flim)

        plt.tight_layout()
        plt.show()
        # Lineouts
        # We've already taken the transpose so y is the first index
        indy = int(beam.Ny/2)
        indx = int(beam.Nx/2)
        x = beam.x
        y = beam.y
        plt.figure(figsize=(16, 4), dpi=150)
        plt.subplot(131)
        plt.plot(x, I[:, indy], label='y')
        plt.plot(y, I[indx, :], '--', label='x')
        plt.legend()
        plt.xlabel(r'$x$ (um)')
        plt.ylabel(r'Intensity ($10^{14}$ W/cm^2)')
        if xlim != None:
            plt.xlim(xlim)

        plt.subplot(132)
        plt.plot(x, np.unwrap(phase[:, indy]), label='x')
        plt.plot(y, np.unwrap(phase[indx, :]), '--', label='y')
        plt.legend()
        plt.xlabel(r'$x$ (um)')
        plt.ylabel(r'Phase (rad)')
        if xlim != None:
            plt.xlim(xlim)

        plt.subplot(133)
        plt.plot(fx, If[:, indy], label='x')
        plt.plot(fy, If[indx, :], '--', label='y')
        plt.legend()
        plt.xlabel(r'$f_x$ (um$^{-1}$)')
        plt.ylabel(r'Intensity (arb unit)')
        if flim != None:
            plt.xlim(flim)

        plt.tight_layout()
        plt.show()
        
        if log == True:
            # Lineouts
            plt.figure(figsize=(16, 4), dpi=150)
            plt.subplot(131)
            plt.plot(x, I[:, indy], label='x')
            plt.plot(y, I[indx, :], '--', label='y')
            plt.legend()
            plt.xlabel(r'$x$ (um)')
            plt.ylabel(r'Intensity ($10^{14}$ W/cm^2)')
            plt.yscale('log')
            if xlim != None:
                plt.xlim(xlim)

            plt.subplot(132)
            plt.plot(x, np.unwrap(phase[:, indy]), label='x')
            plt.plot(y, np.unwrap(phase[indx, :]), '--', label='y')
            plt.legend()
            plt.xlabel(r'$x$ (um)')
            plt.ylabel(r'Phase (rad)')
            plt.yscale('log')
            if xlim != None:
                plt.xlim(xlim)

            plt.subplot(133)
            plt.plot(fx, If[:, indy], label='x')
            plt.plot(fy, If[indx, :], '--', label='y')
            plt.legend()
            plt.xlabel(r'$f_x$ (um$^{-1}$)')
            plt.ylabel(r'Intensity (arb unit)')
            plt.yscale('log')
            if flim != None:
                plt.xlim(flim)

            plt.tight_layout()
            plt.show()
        

class GaussianLaser(Laser):
    """ A laser beam class that creates a Gaussian electric field. 
    
    Parameters
    ----------
    E0 : double
        The peak value of the electric field at the Gaussian waist in GV/m. 
    waist : double
        The spot size of the Gaussian waist.
    z : double
        The position relative to the waist to start the beam at. +z is after
        the waist, -z is before the waist.
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['E0',
                 'waist',
                 'z0'])
        super().__init__(params)
    
    def initialize_field(self):
        """ Create the array to store the electric field values in. 
        
        Fills the field array with the field of a Gaussian pulse.
        """
        k = self.k
        w0 = self.waist
        z0 = self.z0
        E0 = self.E0
        x2 = np.reshape(self.x, (self.Nx, 1))**2
        y2 = np.reshape(self.y, (1, self.Ny))**2
        # Calculate all the parameters for the Gaussian beam
        r2 = x2 + y2
        zr = np.pi*w0**2 / self.lam
        if z0 != 0:
            wz = w0 * np.sqrt(1+(z0/zr)**2)
            Rz = z0 * (1 + (zr/z0)**2)
            psi = np.arctan(z0/zr)
            # Create the Gaussian field
            e = E0 * w0 / wz * np.exp(-r2/wz**2) \
                 * np.exp(1j*(k*z0 + k*r2/(2*Rz) - psi))
        else:
            e = E0 * np.exp(-r2/w0**2)
        super().initialize_field(e)
        
        
class GeneralGaussianLaser(Laser):
    """ A laser beam class that creates a Gaussian electric field. 
    
    The Gaussian beam can be displaced and propagating at an angle.
    
    Parameters
    ----------
    E0 : double
        The peak value of the electric field at the Gaussian waist in GV/m. 
    waist : double
        The spot size of the Gaussian waist.
    z : double
        The position relative to the waist to start the beam at. +z is after
        the waist, -z is before the waist.
    theta : double
        The angle of propagation in degrees, positive is angled upward.
    dx : double
        The displacement of the beam from the center of the grid.
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['E0',
                 'waist',
                 'z0',
                 'theta',
                 'dx'])
        super().__init__(params)
    
    def initialize_field(self):
        """ Create the array to store the electric field values in. 
        
        Fills the field array with the field of a Gaussian pulse.
        """
        k = self.k
        w0 = self.waist
        z0 = self.z0
        E0 = self.E0
        theta = np.radians(self.theta)
        dx = self.dx
        x = self.x[:, None]
        y = self.y[None, :]
        r2 = (x-dx)**2 + y**2
        # Calculate all the parameters for the Gaussian beam
        zr = np.pi*w0**2 / self.lam
        if z0 != 0:
            wz = w0 * np.sqrt(1+(z0/zr)**2)
            Rz = z0 * (1 + (zr/z0)**2)
            psi = np.arctan(z0/zr)
            # Create the Gaussian field
            e = E0 * w0 / wz * np.exp(-r2/wz**2) \
                 * np.exp(1j*(k*z0 + k*r2/(2*Rz) - psi))
        else:
            e = np.array(E0 * np.exp(-r2/w0**2), dtype='complex128')
        e *= np.exp(1j*k*theta*x)
        super().initialize_field(e)


class SuperGaussianLaser(Laser):
    """ A laser beam class that creates a super-Gaussian electric field. 
    
    Parameters
    ----------
    E0 : double
        The peak value of the electric field on the flattop in GV/m. 
    waist : double
        The spot size of the flattop region.
    order : int
        The order of the super Gaussian.
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['E0',
                 'waist',
                 'order'])
        super().__init__(params)
    
    def initialize_field(self):
        """ Create the array to store the electric field values in. 
        
        Fills the field array with the field of a Gaussian pulse.
        """
        w0 = self.waist
        E0 = self.E0
        n = self.order
        x2 = np.reshape(self.x, (self.Nx, 1))**2
        y2 = np.reshape(self.y, (1, self.Ny))**2
        # Calculate all the parameters for the Gaussian beam
        r = np.sqrt(x2 + y2)
        e = E0 * np.exp(-(r/w0)**n)
        super().initialize_field(e)


class GeneralSuperGaussianLaser(Laser):
    """ A laser beam class that creates a super-Gaussian electric field. 
    
    The super-Gaussian beam can be displaced and propagating at an angle.
    
    Parameters
    ----------
    E0 : double
        The peak value of the electric field on the flattop in GV/m. 
    waist : double
        The spot size of the flattop region.
    order : int
        The order of the super Gaussian.
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['E0',
                 'waist',
                 'order',
                 'theta',
                 'dx'])
        super().__init__(params)
    
    def initialize_field(self):
        """ Create the array to store the electric field values in. 
        
        Fills the field array with the field of a Gaussian pulse.
        """
        w02 = self.waist**2
        E0 = self.E0
        n = self.order/2
        k = self.k
        theta = np.radians(self.theta)
        dx = self.dx
        x = self.x[:, None]
        y = self.y[None, :]
        r2 = (x-dx)**2 + y**2
        e = np.array(E0 * np.exp(-(r2/w02)**n), dtype='complex128')
        e *= np.exp(1j*k*theta*x)
        super().initialize_field(e)


class RadialLaser(Laser):
    """ A laser beam with a radially dependent field and periodic phi phase.
    
    The beam has a radially dependent intensity and phase. It can have a
    periodic phase in phi of order n.
    
    Parameters
    ----------
    order : int
        The number of periods of the phase in phi.
    r : array-like
        An array of radial coordinates the electric field is specified at.
    E : array-like
        The radial electric field specified at each element in r.
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['order',
                 'r',
                 'E'])
        super().__init__(params)
    
    def initialize_field(self):
        """ Create the array to store the electric field values in. 
        
        Fills the field array with the field of a Gaussian pulse.
        """
        order = self.order
        x = self.x
        y = self.y
        Nx = self.Nx
        Ny = self.Ny
        e = self.reconstruct_from_cyl(self.r, self.E, x, y)
        # Add the phi dependent phase
        phi = np.zeros((Nx, Ny), dtype='complex128') 
        # Handle when x/y -> ininity
        phi[int(Nx/2), int(Ny/2):] = np.pi/2
        phi[int(Nx/2), :int(Ny/2)] = -np.pi/2
        # Handle the positive x half plane
        sel = np.array(x > 0)
        xp = x[sel]
        xp = np.reshape(xp, (np.size(xp), 1))
        phi[int(Nx/2+1):, :] = np.arctan(y/xp)
        # Handle the negative x half plane
        sel = np.array(x < 0)
        xn = x[sel]
        xn = np.reshape(xn, (np.size(xn), 1))
        phi[:int(Nx/2), :] = np.arctan(y/xn) + np.pi
        e = e * np.exp(1j*phi*order)
        super().initialize_field(e)

