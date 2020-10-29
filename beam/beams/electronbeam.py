#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:49:43 2017

@author: robert
"""

import numpy as np
from beam.beams import beam
from vsim import load as Vload
from vsim import analyze as Vanalyze
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.integrate as Int
from scipy.optimize import curve_fit

class ElectronBeam(beam.Beam):
    """ An electron beam class that stores a collection of macroparticles.
    
    Parameters
    ----------
    N : int
        The number of macroparticles in the beam.
    """
    keys = ['N']
    
    # Initialization functions
    #--------------------------------------------------------------------------
    
    def __init__(self, params):
        super().__init__(params)
        # Create internal variables
        if self.load is False:
            self.initialize_particles()
            self.save_initial()
    
    def initialize_particles(self, ptcls=None):
        """ Create the arrays to store particle positions and momentum in. 
        
        Parameters
        ----------
        ptcls : array-like, optional
            The array of particle position momentum to initialize.
        """
        if ptcls is None:
            self.ptcls = np.zeros((self.N, 6), dtype='double')
        else:
            self.ptcls = np.array(ptcls, dtype='double')
        self.saveInd = 0
        self.z = []
        self.save_ptcls(self.ptcls, 0.0)
    
    def load_beam(self):
        """ Load the beam, specfically load the z-grid and saveInd. """
        self.z = np.load(self.filePre + '_z.npy')
        self.saveInd = len(self.z)
        self.ptcls = self.load_ptcls(self.saveInd - 1)[0]
    
    # Getters and setters
    #--------------------------------------------------------------------------
    
    def get_x(self, ptcls):
        return ptcls[:, 0]
    
    def get_xp(self, ptcls):
        return ptcls[:, 1]
    
    def get_y(self, ptcls):
        return ptcls[:, 2]
    
    def get_yp(self, ptcls):
        return ptcls[:, 3]
    
    def get_z(self, ptcls):
        return ptcls[:, 4]
    
    def get_gamma(self, ptcls):
        return ptcls[:, 5]
    
    #File managment
    #--------------------------------------------------------------------------
    
    def save_ptcls(self, ptcls, z):
        """ Save the current particle distribution to file and advance z. """
        np.save(self.filePre + '_ptcls_' + str(self.saveInd) + '.npy', ptcls)
        self.saveInd += 1
        self.z.append(z)
        self.save_z()
    
    def save_z(self):
        """ Save the z array. """
        np.save(self.filePre + '_z.npy', self.z)
    
    def load_ptcls(self, ind):
        """ Load the particle distribution at the specified index. 
        
        Parameters
        ----------
        ind : int
            The save index to load the field at.
        
        Returns
        -------
        ptcls : array-like
            The particle corrdinates and momenta at the specified index.
        z : double
            The z coordinate of the field.
        """
        ptcls = np.load(self.filePre + '_ptcls_' + str(ind) + '.npy')
        z = self.z[ind]
        return ptcls, z
    
    def get_save_z(self,ind):
        return self.load_ptcls(ind)[1]
    
    # Physics functions
    #--------------------------------------------------------------------------
        
    def propagate(self, z, n):
        """ Propagate the field to an array of z distances. """
        #TODO implement this function
    
    def get_emittance(self, ind, ptcls=None, weights=None):
        """ Calculate the emittance from a particular save file. """
        ptcls = self.load_ptcls(ind)[0]
        x = self.get_x(ptcls)
        xp = self.get_xp(ptcls)
        y = self.get_y(ptcls)
        yp = self.get_yp(ptcls)
        #If weights aren't given, just initialize an array of 1's
        if weights is None:
            weights = np.zeros(len(x))+1
        # Calculate the differences from the average
        dx = x - np.average(x, weights=weights)
        dxp = xp - np.average(xp, weights=weights)
        dy = y - np.average(y, weights=weights)
        dyp = yp - np.average(yp, weights=weights)
        # Calculate the RMS sizes and the correlation
        sigmax2 = np.average(dx**2, weights=weights)
        sigmaxp2 = np.average(dxp**2, weights=weights)
        sigmaxxp = np.average(dx*dxp, weights=weights)
        sigmay2 = np.average(dy**2, weights=weights)
        sigmayp2 = np.average(dyp**2, weights=weights)
        sigmayyp = np.average(dy*dyp, weights=weights)
        # Calculate the emittance
        ex = np.sqrt(sigmax2*sigmaxp2 - sigmaxxp**2)
        ey = np.sqrt(sigmay2*sigmayp2 - sigmayyp**2)
        return ex, ey
    
    def get_gamma_n(self, ind, weights=None):
        """ Calculate the Lorentz factor from a particular save file. """
        ptcls = self.load_ptcls(ind)[0]
        gamma_arr = self.get_gamma(ptcls)
        #If weights aren't given, just initialize an array of 1's
        if weights is None:
            weights = np.zeros(len(gamma_arr))+1
        gamma = np.average(gamma_arr, weights=weights)
        return gamma
    
    def get_emittance_n(self, ind, weights=None):
        """ Calculate the normalized emittance from a particular save file. """
        ex, ey = self.get_emittance(ind, weights=weights)
        gamma = self.get_gamma_n(ind, weights=weights)
        ex = ex*gamma
        ey = ey*gamma
        return ex, ey
    
    def get_sigmar(self, ind, weights=None):
        """ Caclulate the transverse beam size from a particular save file. """
        ptcls = self.load_ptcls(ind)[0]
        x = self.get_x(ptcls)
        y = self.get_y(ptcls)
        #If weights aren't given, just initialize an array of 1's
        if weights is None:
            weights = np.zeros(len(x))+1
        # Calculate the differences from the average
        dx = x - np.average(x, weights=weights)
        dy = y - np.average(y, weights=weights)
        # Calculate the RMS sizes and the correlation
        sigmax = np.sqrt(np.average(dx**2, weights=weights))
        sigmay = np.sqrt(np.average(dy**2, weights=weights))
        return sigmax, sigmay
    
    def get_sigmar_frac(self, ind, weights=None, threshold=1000):
        """ Caclulate the transverse beam size from a particular save file. """
        ptcls = self.load_ptcls(ind)[0]
        #If weights aren't given, just initialize an array of 1's
        x = self.get_x(ptcls)
        y = self.get_y(ptcls)
        if weights is None:
            weights = np.zeros(len(x))+1
            
        total = np.sum(weights)
        inset = np.where(np.sqrt(np.square(x)+np.square(y)) < threshold)
        ptcls = ptcls[inset]; weights = weights[inset]
        print("fraction: ",np.sum(weights)/total*100,"%")
        
        x = self.get_x(ptcls)
        y = self.get_y(ptcls)
        
        # Calculate the differences from the average
        dx = x - np.average(x, weights=weights)
        dy = y - np.average(y, weights=weights)
        # Calculate the RMS sizes and the correlation
        sigmax = np.sqrt(np.average(dx**2, weights=weights))
        sigmay = np.sqrt(np.average(dy**2, weights=weights))
        return sigmax, sigmay
    
    def get_sigmarp(self, ind, weights=None):
        """ Calculate the beam divergence from a particular save file. """
        ptcls = self.load_ptcls(ind)[0]
        xp = self.get_xp(ptcls)
        yp = self.get_yp(ptcls)
        #If weights aren't given, just initialize an array of 1's
        if weights is None:
            weights = np.zeros(len(xp))+1
        # Calculate the differences from the average
        dxp = xp - np.average(xp, weights=weights)
        dyp = yp - np.average(yp, weights=weights)
        # Calculate the RMS sizes and the correlation
        sigmaxp = np.sqrt(np.average(dxp**2, weights=weights))
        sigmayp = np.sqrt(np.average(dyp**2, weights=weights))
        return sigmaxp, sigmayp

    def get_beam_properties(self, ind):
        """ Calculate most of the beam properties from a save file. 
        
        Creates an output dictionary exactly matching Mike's code.
        """
        prop = {}
        
        ptcls = self.load_ptcls(ind)[0]
        x = self.get_x(ptcls)
        xp = self.get_xp(ptcls)
        gamma = np.average(self.get_gamma(ptcls))
        # Calculate the differences from the average
        dx = x - np.average(x)
        dxp = xp - np.average(xp)
        # Calculate the RMS sizes and the correlation
        sigmax2 = np.average(dx**2)
        sigmaxp2 = np.average(dxp**2)
        sigmaxxp = np.average(dx*dxp)
        # Calculate the emittance
        exn = gamma*np.sqrt(sigmax2*sigmaxp2 - sigmaxxp**2)
        prop['x_eps']   = exn
        prop['x']       = np.sqrt(sigmax2)
        prop['xp']      = np.sqrt(sigmaxp2)
        prop['xxp']     = np.sqrt(sigmaxxp)
        prop['x_beta']  = gamma*sigmax2/exn
        prop['x_gamma'] = gamma*sigmaxp2/exn
        prop['x_alpha'] = -gamma*sigmaxxp/exn
        prop['x_phase'] = np.arctan2(2*prop['x_alpha'], 
                          prop['x_gamma']-prop['x_beta'])/2
        return prop
    
    def get_CS_at(self, ind, weights=None):
        """ Find the CS parameters and beam parameters at a given index.
        
        Returns
        -------
        beam : dictionary
            eps_x, eps_y : double
                Geometric emittance in x and y.
            beta_x, beta_y : double
                Beta function in x and y.
            alpha_x, alpha_y : double
                Alpha function in x and y.
            gamma_x, gamma_y : double
                Gamma function in x and y.
            gamma_b : double
                Relativistic gamma of the beam.
            cen_x, cen_y : double
                The transverse beam center in x and y. 
        """
        beam = {}
        
        ptcls = self.load_ptcls(ind)[0]
        x = self.get_x(ptcls)
        xp = self.get_xp(ptcls)
        y = self.get_y(ptcls)
        yp = self.get_yp(ptcls)
        gamma = np.average(self.get_gamma(ptcls), weights=weights)
        # Calculate the differences from the average
        cen_x = np.average(x, weights=weights)
        dx = x - cen_x
        cen_xp = np.average(xp, weights=weights)
        dxp = xp - cen_xp
        cen_y = np.average(y, weights=weights)
        dy = y - cen_y
        cen_yp = np.average(yp, weights=weights)
        dyp = yp - cen_yp
        # Calculate the RMS sizes and the correlation
        sigmax2 = np.average(dx**2, weights=weights)
        sigmaxp2 = np.average(dxp**2, weights=weights)
        sigmaxxp = np.average(dx*dxp, weights=weights)
        sigmay2 = np.average(dy**2, weights=weights)
        sigmayp2 = np.average(dyp**2, weights=weights)
        sigmayyp = np.average(dy*dyp, weights=weights)
        # Calculate the emittance
        ex = np.sqrt(sigmax2*sigmaxp2 - sigmaxxp**2)
        ey = np.sqrt(sigmay2*sigmayp2 - sigmayyp**2)
        beam['eps_x']   = ex
        beam['eps_y']   = ey
        beam['beta_x']  = sigmax2/ex
        beam['beta_y']  = sigmay2/ey
        beam['alpha_x'] = sigmaxxp/ex
        beam['alpha_y'] = sigmayyp/ey
        beam['gamma_x'] = sigmaxp2/ex
        beam['gamma_y'] = sigmayp2/ey
        beam['gamma_b'] = gamma
        beam['cen_x']   = cen_x
        beam['cen_y']   = cen_y
        beam['cen_xp']   = cen_xp
        beam['cen_yp']   = cen_yp
        return beam
    
    def get_CS(self, weights=None):
        """ Return arrays of the CS parameters in each direction.
        
        Returns
        -------
        beam : dictionary
            eps_x, eps_y : array of double
                Geometric emittance in x and y.
            beta_x, beta_y : array of double
                Beta function in x and y.
            alpha_x, alpha_y : array of double
                Alpha function in x and y.
            gamma_x, gamma_y : array of double
                Gamma function in x and y.
            gamma_b : array of double
                Relativistic gamma of the beam.
            cen_x, cen_y : double
                The transverse beam center in x and y. 
        """
        z = self.z
        N = len(z)
        beam = {}
        keys = ['eps_x', 'eps_y', 'beta_x', 'beta_y', 'alpha_x', 'alpha_y',
                'gamma_x', 'gamma_y', 'gamma_b', 'cen_x', 'cen_y', 'cen_xp',
                'cen_yp']
        for key in keys:
            beam[key] = np.zeros(N, dtype='double')
        for i in range(N):
            step = self.get_CS_at(i, weights)
            for key, item in step.items():
                beam[key][i] = item
        return beam
    
    def get_ptcl(self, ind):
        """ Load the 6D phase space for a single particle in the beam.
        
        Parameters
        ----------
        ind : int
            The index of the beam particle.
        
        Returns
        -------
        ptcl : array of double
            An array with 6 rows describing the particles full phase space.
        """
        z = self.z
        N = len(z)
        ptcl = np.zeros((6, N), dtype='double')
        for i in range(N):
            step = self.load_ptcls(i)[0][ind, :]
            ptcl[:, i] = step[:6]
        return ptcl
        
    # Visualization functions
    #--------------------------------------------------------------------------
    
    def plot_current_phase(self, xlim=None, ylim=None):
        """ Plots a scatterplot of the particles in the current beam. """
        self.plot_phase(self.ptcls, self.z[-1], xlim, ylim)
        plt.show()
    
    def plot_phase_at(self, ind):
        """ Plots the particles at a particular z distance.
        
        Parameters
        ----------
        ind : int
            The save index to plot the particles at, see the _z file to find z.
        """
        ptcls, z = self.load_ptcls(ind)
        self.plot_phase(ptcls, z)
        plt.show()
    
    def plot_phase(self, ptcls, z, xlim=None, ylim=None, weights = None):
        """ Create an x-y plot of the particles. """        
        #If weights aren't given, just initialize an array of 1's
        
        #zf = self.get_z(ptcls)
        #weights = zf
        
        if weights is None:
            weights = np.zeros(self.N)+1
        else:
            sort = np.argsort(weights)
            weights = weights[sort]
            ptcls = ptcls[sort]
            

            
        fig = plt.figure(figsize=(10, 4), dpi=150)
        plt.subplot(121)
        sctx = plt.scatter(ptcls[:, 0]*1e6, ptcls[:, 1]*1e3, 1, c=weights)
        plt.title('x phase space')
        plt.xlabel('x (um)')
        plt.ylabel("x' (mrad)")
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.subplot(122)
        scty = plt.scatter(ptcls[:, 2]*1e6, ptcls[:, 3]*1e3, 1, c=weights)
        plt.title('y phase space')
        plt.xlabel('y (um)')
        plt.ylabel("y' (mrad)")
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.tight_layout()
        return fig, sctx, scty
    
    def plot_phase_hist_at(self, ind, fitted = False):
        """ Plots the particles at a particular z distance.  This version uses
        2D binning to create a nicer image
        
        Parameters
        ----------
        ind : int
            The save index to plot the particles at, see the _z file to find z.
        """
        ptcls, z = self.load_ptcls(ind)
        if fitted is False:
            self.plot_phase_hist(ptcls, z, ind)
        else:
            self.plot_phase_hist_fitted(ptcls, z, ind)
    
    def plot_phase_hist(self, ptcls, z, ind, xlim=None, ylim=None, weights = None):
        """ Create an x-y plot of the particles. This version uses 2D binning to
        create a nicer iamge"""        
        #If weights aren't given, just initialize an array of 1's 
        if weights is None:
            weights = np.zeros(self.N)+1
        else:
            sort = np.argsort(weights)
            weights = weights[sort]
            ptcls = ptcls[sort]
    
        sigmay = self.get_sigmar(ind)[0]
        sigmax = self.get_sigmar(ind)[1]
        #For beam sizes ~2.2 ums
        numbins = 101   #bins in 1d hist
        binno = 56      #bins in 2d hist
        xrange = 10#8
        yrange = 0.29
        
        """#For slingshot injected beams
        numbins = 101   #bins in 1d hist
        binno = 56      #bins in 2d hist
        xrange = 100#8
        yrange = 3.0
        """
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.5, 7), dpi = 150)
        plt.rcParams.update({'font.size': 12})
        xhist = plt.subplot(221)
        leftx = plt.hist(ptcls[:,2]*1e6, bins = numbins, weights = weights, color = 'C0', ec='C0')
        x = np.linspace(min(ptcls[:,2]),max(ptcls[:,2]),100)
        fx = max(leftx[0])*np.exp(-1*np.square(x)/2/np.square(sigmax))
        plt.plot(x*1e6, fx, label=r'$\sigma_x=$'+'%s' % float('%.3g' % (sigmax*1e6))+r'$\ \mu m$', color='C1')
        plt.ylabel('Arb. Units')
        plt.ylim(bottom = 0.1)
        plt.ylim(top = max(leftx[0])*1.2)
        plt.xlim([-xrange, xrange])
        plt.legend()
                
        yhist = plt.subplot(222, sharey = xhist)
        lefty = plt.hist(ptcls[:,0]*1e6, bins = numbins, weights = weights, color = 'C0', ec='C0')#, log=True)
        y = np.linspace(min(ptcls[:,0]),max(ptcls[:,0]),100)
        fy = max(lefty[0])*np.exp(-1*np.square(y)/2/np.square(sigmay))
        plt.plot(y*1e6, fy, label=r'$\sigma_y=$'+'%s' % float('%.3g' % (sigmay*1e6))+r'$\ \mu m$', color='C1')
        plt.ylim(bottom = 0.1)
        plt.ylim(top = max(lefty[0])*1.2)
        plt.xlim([-xrange, xrange])
        plt.legend()
    
        xphase = plt.subplot(223, sharex = xhist)
        sctx = plt.hist2d(ptcls[:, 2]*1e6, ptcls[:, 3]*1e3, weights = weights, bins=(binno,binno), cmap=plt.cm.jet, range = [[-xrange,xrange], [-yrange,yrange]], vmax = 100)
        plt.xlabel(r'$x\mathrm{\ [\mu m]}$')
        plt.ylabel(r'$x\rq,\,y\rq\mathrm{\ [mrad]}$')
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        
        yphase = plt.subplot(224, sharex = yhist, sharey = xphase)
        scty = plt.hist2d(ptcls[:, 0]*1e6, ptcls[:, 1]*1e3, weights = weights, bins=(binno,binno), cmap=plt.cm.jet, range = [[-xrange,xrange], [-yrange,yrange]], vmax = 100)
        plt.xlabel(r'$y\mathrm{\ [\mu m]}$')
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
            
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0.05)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-2]], visible=False)
        plt.setp(fig.axes[1].get_yticklabels(), visible = False)
        plt.setp(fig.axes[3].get_yticklabels(), visible = False)
        
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.125, 0.01, 0.376])
        CB = fig.colorbar(sctx[3], cax=cbar_ax)
        CB.set_label("Arb. Units")

        fig.text(.15,.84,"(a)")
        fig.text(.55,.84,"(b)")
        fig.text(.15,.46,"(c)",color='white')
        fig.text(.55,.46,"(d)",color='white')
        #plt.savefig('/home/chris/Desktop/fig_histogram.eps',format='eps',bbox_inches='tight',dpi=100)
        plt.show()
        
        

    def plot_phase_hist_fitted(self, ptcls, z, ind, xlim=None, ylim=None, weights = None):
        #This version is the same as above, but instead of using the calculated RMS for the
        #Gaussian curve here we take fits to the data and output all the stats.
        
        #If weights aren't given, just initialize an array of 1's 
        if weights is None:
            weights = np.zeros(self.N)+1
        else:
            sort = np.argsort(weights)
            weights = weights[sort]
            ptcls = ptcls[sort]
        
        numbins = 101   #bins in 1d hist
        binno = 56      #bins in 2d hist
        xrange = 8
        yrange = 0.29
        
        """
        #For slingshot injected beams
        numbins = 101   #bins in 1d hist
        binno = 56      #bins in 2d hist
        xrange = 10#8
        yrange = 50.0
        """
        ##Below is same stuff, but now we want to calculate some offsets in the 
        ## transverse dimensions and do actual fits to find the true sigmas
        
        parts_x  = ptcls[:,2]*1e6
        parts_xp = ptcls[:,3]*1e3
        parts_y  = ptcls[:,0]*1e6
        parts_yp = ptcls[:,1]*1e3
        x0  = np.average(parts_x, weights = weights)
        xp0 = np.average(parts_xp,weights = weights)
        y0  = np.average(parts_y, weights = weights)
        yp0 = np.average(parts_yp,weights = weights)
        print('x0',x0)
        print('xp0',xp0)
        print('y0',y0)
        print('yp0',yp0); print()
        sigx = np.sqrt(np.average((parts_x-x0)**2, weights=weights))
        sigxp = np.sqrt(np.average((parts_xp-xp0)**2, weights=weights))
        sigy = np.sqrt(np.average((parts_y-y0)**2, weights=weights))
        sigyp = np.sqrt(np.average((parts_yp-yp0)**2, weights=weights))
        print('sigx',sigx)
        print('sigxp',sigxp)
        print('sigy',sigy)
        print('sigyp',sigyp); print()
        sigy_alt = np.sqrt(np.average((parts_y)**2, weights=weights))
        print('alt sigy',sigy_alt); print()
        
        ## Below I am fitting some Gaussian functions
        leftx = plt.hist(ptcls[:,2]*1e6, bins = numbins, weights = weights, color = 'C0', ec='C0')
        xhist_data = np.array(leftx[0])
        xhist_bins = np.array(leftx[1])
        xbins_cent = (xhist_bins[:-1]+xhist_bins[1:])/2
        
        # Define model function to be used to fit to the data above:
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))
        
        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [1., 0., 1.]
        
        xcoeff, var_matrix = curve_fit(gauss, xbins_cent, xhist_data, p0=p0)
        
        # Get the fitted curve
        hist_fit = gauss(xbins_cent, *xcoeff)
        
        plt.plot(xbins_cent, xhist_data, label='Test data')
        plt.plot(xbins_cent, hist_fit, label='Fitted data')
        
        I = np.sum(xhist_data)
        Isize = xbins_cent[5]-xbins_cent[4]
        factor = I/0.5*Isize #(nC/um)^-1
        #print(Isize)
        #print("test: ", np.sum(xhist_data)/factor)
        
        sctx = plt.hist2d(ptcls[:, 2]*1e6, ptcls[:, 3]*1e3, weights = weights, bins=(binno,binno), cmap=plt.cm.jet, range = [[-xrange,xrange], [-yrange,yrange]], vmax = 100)
        plt.show()
        I2 = np.sum(sctx[0])
        I2size = (sctx[1][2]-sctx[1][1])*(sctx[2][2]-sctx[2][1])
        factor2 = I2/0.5*I2size #(nC/um^2-rad)^-1
        #print(factor2)
        
        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        print('x Fitted mean = ', xcoeff[1])
        print('x Fitted standard deviation = ', xcoeff[2])
        print('xmax = ', xcoeff[0])
        plt.title("x data from Gaussian fit")
        plt.show()
        
        lefty = plt.hist(ptcls[:,0]*1e6, bins = numbins, weights = weights, color = 'C0', ec='C0')
        yhist_data = np.array(lefty[0])
        yhist_bins = np.array(lefty[1])
        ybins_cent = (yhist_bins[:-1]+yhist_bins[1:])/2
        
        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [1., 0., 1.]
        
        ycoeff, var_matrix = curve_fit(gauss, ybins_cent, yhist_data, p0=p0)
        
        # Get the fitted curve
        hist_fit = gauss(ybins_cent, *ycoeff)
        
        plt.plot(ybins_cent, yhist_data, label='Test data')
        plt.plot(ybins_cent, hist_fit, label='Fitted data')
        
        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        print('y Fitted mean = ', ycoeff[1])
        print('y Fitted standard deviation = ', ycoeff[2])
        print('ymax = ', ycoeff[0])
        plt.title("y data from Gaussian fit")
        plt.show()
        
        sigx = np.abs(xcoeff[2])
        sigy = np.abs(ycoeff[2])
        x0   = xcoeff[1]
        y0   = ycoeff[1]
        peakx= xcoeff[0]
        peaky= ycoeff[0]
        
        ##Ill comment out the lines for just plotting normal sigmas w/out correction
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.5, 7), dpi = 150)
        plt.rcParams.update({'font.size': 12})
        xhist = plt.subplot(221)
        leftx = plt.hist(ptcls[:,2]*1e6, bins = numbins, weights = weights/factor, color = 'C0', ec='C0')
        
        x = np.linspace(min(ptcls[:,2]),max(ptcls[:,2]),100)
        fx = peakx*np.exp(-1*np.square(x-x0/1e6)/2/np.square(sigx/1e6))
        
        leftx = plt.hist(ptcls[:,2]*1e6, bins = numbins, weights = weights/factor, color = 'C0', ec='C0')
        plt.plot(x*1e6, fx/factor, label=r'$\sigma_x=$'+'%s' % float('%.3g' % (sigx))+r'$\ \mu m$', color='C1')
        plt.ylabel(r'Charge Density ($\mathrm{nC/ \mu m}$)')
        plt.ylim(bottom = 0.0005)
        plt.ylim(top = max(leftx[0])*1.25)
        plt.xlim([-xrange, xrange])
        plt.legend()
                
        yhist = plt.subplot(222, sharey = xhist)
        lefty = plt.hist(ptcls[:,0]*1e6, bins = numbins, weights = weights/factor, color = 'C0', ec='C0')#, log=True)
        y = np.linspace(min(ptcls[:,0]),max(ptcls[:,0]),100)
        fy = peaky*np.exp(-1*np.square(y-y0/1e6)/2/np.square(sigy/1e6))
        plt.plot(y*1e6, fy/factor, label=r'$\sigma_y=$'+'%s' % float('%.3g' % (sigy))+r'$\ \mu m$', color='C1')
        plt.ylim(bottom = 0.0005)
        plt.ylim(top = max(lefty[0])*1.25)
        plt.xlim([-xrange, xrange])
        plt.legend()
    
        xphase = plt.subplot(223, sharex = xhist)
        sctx = plt.hist2d(ptcls[:, 2]*1e6, ptcls[:, 3]*1e3, weights = weights/factor2, bins=(binno,binno), cmap=plt.cm.jet, range = [[-xrange,xrange], [-yrange,yrange]], vmax = 100/factor2)
        plt.xlabel(r'$x\mathrm{\ (\mu m)}$')
        plt.ylabel(r'$x\rq,\,y\rq\mathrm{\ (mrad)}$')
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        
        yphase = plt.subplot(224, sharex = yhist, sharey = xphase)
        scty = plt.hist2d(ptcls[:, 0]*1e6, ptcls[:, 1]*1e3, weights = weights/factor2, bins=(binno,binno), cmap=plt.cm.jet, range = [[-xrange,xrange], [-yrange,yrange]], vmax = 100/factor2)
        plt.xlabel(r'$y\mathrm{\ (\mu m)}$')
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
            
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0.05)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-2]], visible=False)
        plt.setp(fig.axes[1].get_yticklabels(), visible = False)
        plt.setp(fig.axes[3].get_yticklabels(), visible = False)
        
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.125, 0.01, 0.376])
        CB = fig.colorbar(sctx[3], cax=cbar_ax)
        CB.set_label(r'Charge Density ($\mathrm{nC/ \mu m-mrad}$)')

        fig.text(.15,.84,"(a)")
        fig.text(.55,.84,"(b)")
        fig.text(.15,.46,"(c)",color='white')
        fig.text(.55,.46,"(d)",color='white')
        
        #plt.savefig('/home/chris/Desktop/fig6.eps',format='eps',bbox_inches='tight',dpi=150)
        plt.show()

    def plot_hist_at(self, ind):
        ptcls, z = self.load_ptcls(ind)
        self.plot_hist(ptcls, z, ind)
    
    def plot_hist(self, ptcls, z, ind, weights = None):
        if weights is None:
            weights = np.zeros(self.N)+1
        else:
            sort = np.argsort(weights)
            weights = weights[sort]
            ptcls = ptcls[sort]
        
        sigmay = self.get_sigmar(ind)[0]
        sigmax = self.get_sigmar(ind)[1]
        #There used to be many different fits, but now only the Gauss+Gauss remains.
        # If you want to implement more, use the general procudure below with
        # functions at the bottom of this script.
        
        numbins = 101

        ##These two are x and y side by side.
        plt.figure(figsize=(10, 4), dpi=150)
        plt.subplot(121)
        xhist = plt.hist(ptcls[:,2]*1e6, bins = numbins, weights = weights, log=True)  
        x = np.linspace(min(ptcls[:,2]),max(ptcls[:,2]),100)
        fx = max(xhist[0])*np.exp(-1*np.square(x)/2/np.square(sigmax))
        plt.plot(x*1e6, fx, label=r'$\sigma_x=$'+str(sigmax*1e6)+r'$\ \mu m$')
        plt.xlabel(r'$x\mathrm{\ [\mu m]}$')
        plt.ylabel('Counts')
        plt.ylim(bottom = 0.1)
        plt.ylim(top = max(xhist[0])*6)
        plt.legend()
        
        plt.subplot(122)
        yhist = plt.hist(ptcls[:,0]*1e6, bins = numbins, weights = weights, log=True)
        y = np.linspace(min(ptcls[:,0]),max(ptcls[:,0]),100)
        fy = max(yhist[0])*np.exp(-1*np.square(y)/2/np.square(sigmay))
        plt.plot(y*1e6, fy, label=r'$\sigma_y=$'+str(sigmay*1e6)+r'$\ \mu m$')
        plt.xlabel(r'$y\mathrm{\ [\mu m]}$')
        plt.ylabel('Counts')
        plt.ylim(bottom = 0.1)
        plt.ylim(top = max(yhist[0])*6)
        plt.legend(); plt.show()
        
        #With the data from left we now want to try a Gauss + Gauss fit
        #At some point I will move this to its own function 
        
        xsel = True
        if xsel is True:
            histdata = xhist
            indnum = 2
            signum = sigmax
            label = 'x'
        else:
            histdata = yhist
            indnum = 0
            signum = sigmay
            label = 'y'
        
        wdata = np.array(histdata[0])
        rdata = np.array(histdata[1])*1e-6
        rdata = rdata[0:-1]+.5*(rdata[1]-rdata[0])
        
        out = 2
        if out==1:#Gaussian
            p = FitDataSomething(wdata, rdata, GaussPlusGauss, [signum/2, signum, max(histdata[0]), max(histdata[0])/100])
            print("G+G: ",p)
            llabel = "G+G: "+r'$\sigma_1=$'+("%0.2f"%(np.abs(p[0])*1e6))+r'$\ \mu m$'+" & "+r'$\sigma_2=$'+("%0.2f"%(np.abs(p[1])*1e6))+r'$\ \mu m$'
            fxg = GaussPlusGauss(p,x)
        elif out==2:#Exponential
            p = FitDataSomething(wdata, rdata, GaussPlusExp, [signum/2, signum, max(histdata[0]), max(histdata[0])/100])
            print("G+E: ",p)
            llabel = "Gauss+Exponential: "+r'$\sigma_{Gauss}=$'+("%0.2f"%(np.abs(p[0])*1e6))+r'$\ \mu m$'
            fxg = GaussPlusExp(p,x)
        
        plt.figure(figsize=(10, 4), dpi=150)
        plt.subplot(121)
        left = plt.hist(ptcls[:,indnum]*1e6, bins = numbins, weights = weights, log=True)
        plt.plot(x*1e6, fxg, label=llabel)
        plt.plot(x*1e6, Gauss([p[0],p[2]],x), 'k--')
        if out == 1:
            plt.plot(x*1e6, Gauss([p[1],p[3]],x), 'c--')
        elif out == 2:
            plt.plot(x*1e6, ExpFunc([p[1],p[3]],x), 'c--')
        plt.xlabel(label+r'$\mathrm{\ [\mu m]}$')
        plt.ylabel('Counts')
        plt.ylim(bottom = 0.1)
        plt.ylim(top = max(left[0])*6)
        plt.legend()
        
        plt.subplot(122)
        right = plt.hist(ptcls[:,indnum]*1e6, bins = numbins, weights = weights, log=False)
        plt.plot(x*1e6, fxg, label=llabel)
        plt.plot(x*1e6, Gauss([p[0],p[2]],x), 'k--')
        if out == 1:
            plt.plot(x*1e6, Gauss([p[1],p[3]],x), 'c--')
        elif out == 2:
            plt.plot(x*1e6, ExpFunc([p[1],p[3]],x), 'c--')
        plt.xlabel(label+r'$\mathrm{\ [\mu m]}$')
        plt.ylim(bottom = 0.1)
        plt.ylim(top = max(right[0])*1.2)
        plt.legend(); plt.show()
        
        if out == 1:
            GaussPlusGauss_Percent(p)
        elif out == 2:
            GaussPlusExp_Percent(p)
        
        return

class GaussianElectronBeam(ElectronBeam):
    """ A electron beam with a Gaussian transverse profile. 
    
    Parameters
    ----------
    gamma : double
        The relativistic factor of the beam.
    emittance : double
        The normalized emittance of the beam in m*rad.
    betax : double
        The beta function in the x-direction at z=0, in m.
    betay : double
        The beta function in the y direction at z=0, in m.
    alphax : double
        The alpha parameter of the beam in the x-direction at z=0.
    alphay : double
        The alpha parameter of the beam in the y-direction at z=0.
    sigmaz : double
        The RMS size of the beam in z in m.
    dE : double
        The energy spread of the beam as a fraction, 0.01 = +-1% energy spread.
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['gamma',
                 'emittance',
                 'betax',
                 'betay',
                 'alphax',
                 'alphay',
                 'sigmaz',
                 'dE'])
        super().__init__(params)
        
    def action_angle_distribution(self):
        """ Initialize particles in action-angle coordinates. 
        
        Returns
        -------
        ux : array of double
            Particle positions in ux.
        vx : array of double
            Particle positions in vx.
        uy : array of double
            Particle positions in uy.
        vy : array of double
            Particle positions in vy.
        """
        N = self.N
        gamma = self.gamma
        emittance = self.emittance
        # Calculate arrays of random numbers
        x1r = np.random.uniform(0, 1, N)
        x2r = np.random.uniform(0, 1, N)
        y1r = np.random.uniform(0, 1, N)
        y2r = np.random.uniform(0, 1, N)
        # Choose the particles from a distribution
        Jx = -emittance * np.log(x1r) / gamma
        Jy = -emittance * np.log(y1r) / gamma
        phix = 2*np.pi*x2r
        phiy = 2*np.pi*y2r
        ux = np.sqrt(2*Jx)*np.cos(phix)
        vx = -np.sqrt(2*Jx)*np.sin(phix)
        uy = np.sqrt(2*Jy)*np.cos(phiy)
        vy = -np.sqrt(2*Jy)*np.sin(phiy)
        return ux, vx, uy, vy
    
    def initialize_particles(self, offset_x=0.0, offset_y=0.0, offset_xp=0.0, offset_yp=0.0):
        """ Initialize the particles in a 6D distribution. """
        N = self.N
        gamma = self.gamma
        betax = self.betax
        betay = self.betay
        ptcls = np.zeros((N, 6), dtype='double')
        ux, vx, uy, vy = self.action_angle_distribution()
        # Calculate the coordinates
        ptcls[:, 0] = ux*np.sqrt(betax) + offset_x
        ptcls[:, 1] = (vx-self.alphax*ux) / np.sqrt(betax) + offset_xp
        ptcls[:, 2] = uy*np.sqrt(betay) + offset_y
        ptcls[:, 3] = (vy-self.alphay*uy) / np.sqrt(betay) + offset_yp
        ptcls[:, 4] = np.random.normal(0.0, self.sigmaz, N)
        ptcls[:, 5] = gamma * (1 + self.dE*np.random.uniform(-1, 1, N))
        super().initialize_particles(ptcls) 

class OffsetGaussianElectronBeam(GaussianElectronBeam):
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend(
                ['offset_x',
                 'offset_y',
                 'offset_xp',
                 'offset_yp'])
        super().__init__(params)
        
    def initialize_particles(self):
        """ Initialize the particles in a 6D distribution. """
        super().initialize_particles(self.offset_x, self.offset_y, self.offset_xp, self.offset_yp) 

class VorpalElectronBeam(ElectronBeam):
    """ A electron beam imported from VSim, includes weights. 
    
    Parameters
    ----------
    filename : string
        Filename of h5 file to load data from
    thresh : double
        Lowest particle weight to load, set to 0 to load all
    """
    
    def __init__(self, params):
        self.keys = self.keys.copy()
        self.keys.extend([
                'filename',
                'thresh',
                'minz',
                'maxz'])
        super().__init__(params)
    
    def initialize_particles(self):
        """ Initialize the particles in a 6D distribution. """
        #Given a filename, parse the info
        # 'Drive_Witness_Ramps_WitnessBeam_2.h5'
        filename = self.filename
        thresh = self.thresh
        minz = self.minz
        maxz = self.maxz
        
        file = filename.split("/")[-1]
        strlist = file.split("_")
        dumpInd = int(strlist[-1].split(".")[0])
        species = strlist[-2]
        simname = "_".join(strlist[:-2])
        
        data = Vload.get_species_data(filename, species)
        dim = int(data.attrs['numSpatialDims'])
        
        y = np.array(Vanalyze.get_y(data,dim)) #tran
        uy = np.array(Vanalyze.get_uy(data,dim))
        z = np.array(Vanalyze.get_z(data,dim)) #tran
        uz = np.array(Vanalyze.get_uz(data,dim))
        x = np.array(Vanalyze.get_x(data,dim)) #long
        gamma = np.array(Vanalyze.get_ptc_gamma(data))
        weights = np.array(Vanalyze.get_weights(data))
        
        ux = np.array(Vanalyze.get_ux(data,dim))
        uy = uy/ux
        uz = uz/ux
        
        x = x-np.average(x, weights=weights)#recenter to x=-0
        
        if minz > -np.inf:
            zset = np.array(np.where(x > minz)[0])
            y = y[zset]
            uy = uy[zset]
            z = z[zset]
            uz = uz[zset]
            x = x[zset]
            gamma = gamma[zset]
            weights = weights[zset]
            
        if maxz < np.inf:
            zset = np.array(np.where(x < maxz)[0])
            y = y[zset]
            uy = uy[zset]
            z = z[zset]
            uz = uz[zset]
            x = x[zset]
            gamma = gamma[zset]
            weights = weights[zset]
        
        threshset = np.array(np.where(weights > thresh)[0])
        N = len(threshset)
        self.N = N
        
        ptcls = np.zeros((N, 7), dtype='double')
        ptcls[:, 0] = y[threshset]
        ptcls[:, 1] = uy[threshset]
        ptcls[:, 2] = z[threshset]
        ptcls[:, 3] = uz[threshset]
        ptcls[:, 4] = x[threshset]
        ptcls[:, 5] = gamma[threshset]
        ptcls[:, 6] = weights[threshset]
        
        super().initialize_particles(ptcls)
    
    def get_weights(self, ptcls):
        return ptcls[:, 6]

    def get_emittance(self, ind):
        ptcls = self.load_ptcls(ind)[0]
        weights = self.get_weights(ptcls)
        return super().get_emittance(ind, weights)
        
    def get_gamma_n(self, ind):
        ptcls = self.load_ptcls(ind)[0]
        weights = self.get_weights(ptcls)
        return super().get_gamma_n(ind, weights)

    def get_emittance_n(self, ind):
        """ Calculate the normalized emittance from a particular save file. """
        ptcls = self.load_ptcls(ind)[0]
        weights = self.get_weights(ptcls)
        ex, ey = super().get_emittance(ind, weights=weights)
        gamma = super().get_gamma_n(ind, weights=weights)
        ex = ex*gamma
        ey = ey*gamma
        return ex, ey

    def get_sigmar(self, ind):
        ptcls = self.load_ptcls(ind)[0]
        weights = self.get_weights(ptcls)
        return super().get_sigmar(ind, weights)

    def get_sigmar_frac(self, ind, threshold):
        ptcls = self.load_ptcls(ind)[0]
        weights = self.get_weights(ptcls)
        return super().get_sigmar_frac(ind, weights, threshold)

    def get_sigmarp(self, ind):
        ptcls = self.load_ptcls(ind)[0]
        weights = self.get_weights(ptcls)
        return super().get_sigmarp(ind, weights)
    
    def plot_current_phase(self, xlim=None, ylim=None):
        weights = self.get_weights(self.ptcls)
        super().plot_phase(self.ptcls, self.z[-1], xlim, ylim, weights=weights)
        plt.show()
    
    def plot_phase_at(self, ind):
        ptcls, z = self.load_ptcls(ind)
        weights = self.get_weights(ptcls)
        super().plot_phase(ptcls, z, weights=weights)
        plt.show()
    
    def plot_phase(self, ptcls, z, xlim=None, ylim=None):
        weights = self.get_weights(ptcls)
        super().plot_phase(ptcls, z, xlim, ylim, weights)
        
    def plot_phase_hist_at(self, ind, fitted = False):
        ptcls, z = self.load_ptcls(ind)
        weights = self.get_weights(ptcls)
        if fitted is False:
            super().plot_phase_hist(ptcls, z, ind, weights=weights)
        else:
            super().plot_phase_hist_fitted(ptcls, z, ind, weights=weights)
        plt.show()
    
    def plot_phase_hist(self, ptcls, z, ind, xlim=None, ylim=None):
        weights = self.get_weights(ptcls)
        super().plot_phase_hist(ptcls, z, ind, xlim, ylim, weights)
        
    def plot_phase_hist_fitted(self, ptcls, z, ind, xlim=None, ylim=None):
        weights = self.get_weights(ptcls)
        super().plot_phase_hist_fitted(ptcls, z, ind, xlim, ylim, weights)
        
    def plot_hist_at(self, ind):
        ptcls, z = self.load_ptcls(ind)
        weights = self.get_weights(ptcls)
        super().plot_hist(ptcls, z, ind, weights=weights)
        plt.show()
    
    def plot_hist(self, ptcls, z, ind):
        weights = self.get_weights(ptcls)
        super().plot_hist(ptcls, z, weights)
    
    def get_emittance_n_zcond(self, ind, minz, maxz):
        """ Calculate the normalized emittance from a particular save file. """
        ptcls = self.load_ptcls(ind)[0]
        weights = self.get_weights(ptcls)
        ex, ey = self.get_emittance_zcond(ind, minz, maxz, weights=weights)
        gamma = super().get_gamma_n(ind, weights=weights)
        ex = ex*gamma
        ey = ey*gamma
        return ex, ey    
    
    def get_emittance_zcond(self, ind, minz, maxz, weights, ptcls=None):
        """ Calculate the emittance from a particular save file. """
        ptcls = self.load_ptcls(ind)[0]
        x = self.get_x(ptcls)
        xp = self.get_xp(ptcls)
        y = self.get_y(ptcls)
        yp = self.get_yp(ptcls)
        #If weights aren't given, just initialize an array of 1's
            
        z = self.get_z(ptcls)
        zset = np.array(np.where((z > minz))[0])
        x = x[zset]
        xp = xp[zset]
        y = y[zset]
        yp = yp[zset]
        weights = weights[zset]
        z = z[zset]
        
        zset = np.array(np.where((z < maxz))[0])
        x = x[zset]
        xp = xp[zset]
        y = y[zset]
        yp = yp[zset]
        weights = weights[zset]
        
        # Calculate the differences from the average
        dx = x - np.average(x, weights=weights)
        dxp = xp - np.average(xp, weights=weights)
        dy = y - np.average(y, weights=weights)
        dyp = yp - np.average(yp, weights=weights)
        # Calculate the RMS sizes and the correlation
        sigmax2 = np.average(dx**2, weights=weights)
        sigmaxp2 = np.average(dxp**2, weights=weights)
        sigmaxxp = np.average(dx*dxp, weights=weights)
        sigmay2 = np.average(dy**2, weights=weights)
        sigmayp2 = np.average(dyp**2, weights=weights)
        sigmayyp = np.average(dy*dyp, weights=weights)
        # Calculate the emittance
        ex = np.sqrt(sigmax2*sigmaxp2 - sigmaxxp**2)
        ey = np.sqrt(sigmay2*sigmayp2 - sigmayyp**2)
        return ex, ey
    
#p[sig,gam,n0]
def Voigt(p, x):
    lorentz = (p[1]/(np.square(x)+np.square(p[1])))/np.pi
    gauss = np.exp(-.5*np.square(x)/np.square(p[0]))/p[0]/np.sqrt(2*np.pi)
    return p[2]*lorentz*gauss

def GaussPlusLorentz(p, x):
    lorentz = (p[1]/(np.square(x)+np.square(p[1])))/np.pi
    gauss = np.exp(-.5*np.square(x)/np.square(p[0]))
    return p[2]*gauss+p[3]*lorentz

def GaussPlusGauss(p, x):
    gauss1 = np.exp(-.5*np.square(x)/np.square(p[0]))
    gauss2 = np.exp(-.5*np.square(x)/np.square(p[1]))
    return p[2]*gauss1 + p[3]*gauss2

def GaussPlusExp(p, x):
    gauss1 = np.exp(-.5*np.square(x)/np.square(p[0]))
    exp2 = ExpFunc([p[1],p[3]],x)
    return p[2]*gauss1 + exp2

def GaussTimesGauss(p, x):
    gauss1 = np.exp(-.5*np.square(x)/np.square(p[0]))
    gauss2 = np.exp(-.5*np.square(x)/np.square(p[1]))
    return p[2]*gauss1 * gauss2

#p[gam,B]
def Lorentz(p,x):
    return p[1]*(p[0]/(np.square(x)+np.square(p[0])))/np.pi

#p[sig,A]
def Gauss(p, x):
    return p[1]*np.exp(-.5*np.square(x)/np.square(p[0]))

#p[delta,A]
#def ExpFunc(p, x):
#    return p[1]*np.exp(-1*np.abs(x)/p[0])

#p[delta,A]
def ExpFunc(p, x):
    return p[1]*np.exp(-1*np.abs(x)/p[0])

def FitDataSomething(data, axis, function, guess = [0.,0.,0.]):
    errfunc = lambda p, x, y: function(p, x) - y
    p0 = guess
    p1, success = optimize.leastsq(errfunc,p0[:], args=(axis, data))
    """
    plt.plot(axis, data, label=datlabel)
    #plt.plot(axis, function(guess,axis), label="Guess "+ function.__name__ +" profile")
    plt.plot(axis, function(p1,axis), label="Fitted "+ function.__name__ +" profile")
    plt.title("Comparison of data with "+ function.__name__ +" profile")
    plt.legend(); plt.grid(); plt.show()
    """
    return p1

#Given parameters for the Gauss + Gauss fit, what percentage is particles are in each Gaussian
def GaussPlusGauss_Percent(p):
    pi = [np.abs(p[0]), np.abs(p[2])]
    po = [np.abs(p[1]), np.abs(p[3])]
    inner = Int.quad(lambda x: Gauss(pi, x), -5*pi[0], 5*pi[0])[0]
    outer = Int.quad(lambda x: Gauss(po, x), -5*po[0], 5*po[0])[0]
    print(inner, outer)
    print("Inner: ",inner/(inner+outer)*100,"%")
    print("Outer: ",outer/(inner+outer)*100,"%")
    return    

def GaussPlusExp_Percent(p):
    pi = [np.abs(p[0]), np.abs(p[2])]
    po = [np.abs(p[1]), np.abs(p[3])]
    inner = Int.quad(lambda x: Gauss(pi, x), -5*pi[0], 5*pi[0])[0]
    outer = Int.quad(lambda x: ExpFunc(po, x), -5*po[0], 5*po[0])[0]
    print(inner, outer)
    print("Inner: ",inner/(inner+outer)*100,"%")
    print("Outer: ",outer/(inner+outer)*100,"%")
    return    
    
    
    
    
    
    
    
    
    
    
    
    