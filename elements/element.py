#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:08:32 2017

@author: robert
"""

import os
import glob
import numpy as np
from scipy.interpolate import interp1d


class Element:
    """ The base class for elements.
    
    Implements base methods to check class construction.
    """
    keys = ['name',
            'path',
            'load']
    
    def __init__(self, params):
        self.params = params
        self.check_params(params)
        self.params_to_attrs(params)
        # Create a folder to store the beam data in
        self.dirName = dirName = self.path + 'elements/element_' + self.name \
                                 + '/'
        self.filePre = dirName + self.name
        if self.load is True:
            self.load_params()
            self.load_element()
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
    
    def load_element(self):
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
        dataOfR = interp1d(r, data, bounds_error=False, fill_value=0.0)
        return dataOfR(np.sqrt(x[:, None]**2 + y[None, :]**2))
    