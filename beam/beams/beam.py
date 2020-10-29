#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:04:25 2017

@author: robert
"""

import os
import glob
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.integrate import simps


class Beam:
    """ The base class for beams.
    
    Implements base methods to check class construction.
    """
    keys = ['name',
            'path',
            'load']
    
    # Initialization functions
    #--------------------------------------------------------------------------
    
    def __init__(self, params):
        self.params = params
        self.check_params(params)
        self.params_to_attrs(params)
        # Create a folder to store the beam data in
        self.dirName = dirName = self.path + 'beams/beam_' + self.name + '/'
        self.filePre = dirName + self.name
        if self.load is True:
            self.load_params()
            self.load_beam()
        elif self.load is False:
            if not os.path.exists(dirName):
                os.makedirs(dirName)
            self.clear_dir()
    
    def check_params(self, params):
        """ Check to ensure all required keys are in the params dictionary. """
        for key in self.keys:
            if key not in params:
                raise ValueError('The params object has no key: %s' % key)
    
    def params_to_attrs(self, params):
        """ Add all params as attributes of the class. """
        for key in params:
            setattr(self, key, params[key])
    
    def load_beam(self):
        """ Prototype for child specific loading setup. """
    
    # File managment
    #--------------------------------------------------------------------------
    
    def save_initial(self):
        """ Save the initial params object. """
        np.save(self.filePre + '_params.npy', self.params)    
    
    def clear_dir(self):
        """ Clear all .npy files from the beam directory. """ 
        filelist = glob.glob(self.dirName + '*.npy')
        for f in filelist:
            os.remove(f)
            
    def load_params(self):
        """ Load the params from a saved file"""
        self.params = np.load(self.filePre + '_params.npy').item()
        self.check_params(self.params)
        self.params_to_attrs(self.params)
        self.load = True
    
    # Physics functions
    #--------------------------------------------------------------------------
    
    def intensity_from_field(self, e, n=1):
        """ Calculates the time averaged intensity from the complex field. 
        
        This function goes from GV/m -> 10^14W/cm^2
        """
        return 1.32721e-3 * n * abs(e)**2
    
    def total_cyl_power(self, r, I):
        """ Calculates the total power in the beam.
        
        The functions takes an array of intensity values in 10^14W/cm^2 and an
        array of radii in um and returns power in TW.
        """
        r = r*1e-4 # Convert to cm
        return 2*np.pi*simps(r*I, r)*100
    
    # Visualization functions
    #--------------------------------------------------------------------------
    
    def prep_data(self, data):
        """ Restructures data so that imshow displays it properly. 
        
        If data starts in (x, y) format, this will display it on the correct 
        axis.
        """
        return np.flipud(np.transpose(data))
    
    def reconstruct_from_cyl(self, r, data, x, y):
        """ Create a 2D field from a radial slice of a cylindircal field. """
        dataOfR = interp1d(r, data, bounds_error=False, fill_value=0.0, kind='cubic')
        return dataOfR(np.sqrt(x[:, None]**2 + y[None, :]**2))
    
    def reconstruct_from_cyl_beam(self, beam):
        """ Create a 2D field from a cyclindircally symmetric input beam. """
        r = beam.x
        if len(np.shape(beam.e)) == 2:
            data = np.array(beam.e[:, int(beam.Ny/2)])
        if len(np.shape(beam.e)) == 3:
            data = np.array(beam.e[int(beam.Nt/2),: , int(beam.Ny/2)])
        x = self.x
        y = self.y
        return self.reconstruct_from_cyl(r, data, x, y)
    
    def rescale_field(self, beam1, beam2):
        """ Rescale a laser beam from one grid size to another laser field. """
        e_r = interp2d(beam1.x, beam1.y, beam1.e.real, 'cubic', bounds_error=False, fill_value=0.0)
        e_i = interp2d(beam1.x, beam1.y, beam1.e.imag, 'cubic', bounds_error=False, fill_value=0.0)
        E_r = e_r(beam2.x, beam2.y)
        E_i = e_i(beam2.x, beam2.y)
        E = E_r + 1j*E_i
        return E
    
    