#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:56:11 2017

@author: robert
"""

import numpy as np
import beam.beams as beams
import beam.elements as elements
import beam.calc.plasma as pcalc
import beam.calc.laser as lcalc
import beam.calc.electron as ecalc


def pulse_plasma(pulse, plasma):
    """ Propagates a pulse through a gas, ionizing and refracting as it goes.
    
    Parameters
    ----------
    pulse : Pulse class
        The laser pulse to propagate through the plasma.
    plasma : Plasma class
        The gas to propagate the laser pulse through.
    """
    pulse.e = np.array(pcalc.plasma_refraction(pulse.e, pulse.x, pulse.y,
                      plasma.z, pulse.t, pulse.lam, plasma.n0, pulse.z[-1],
                      pulse.fft, pulse.ifft, pulse.save_field, 
                      plasma.save_plasma_density, plasma.atom, 
                      plasma.load_num_den, plasma.load_plasma_den,
                      pulse.threads))


def pulse_plasma_energy(pulse, plasma, temp=0.0, n2=0.0, ionization='adk'):
    """ Propagates a pulse through a gas, ionizing and refracting as it goes.
    
    Parameters
    ----------
    pulse : Pulse class
        The laser pulse to propagate through the plasma.
    plasma : Plasma class
        The gas to propagate the laser pulse through.
    temp : double, optional
        Temperature of the plasma in eV.
    n2 : double, optional
        The nonlinear index of refraction at atmospheric pressure. In cm^2/W.
    ionization : string, optional
        The ionization model, options are:
            adk
            lithium
    """
    pulse.e = np.array(pcalc.plasma_refraction_energy(pulse.e, pulse.x, pulse.y,
                      plasma.z, pulse.t, pulse.lam, plasma.n0, pulse.z[-1],
                      pulse.fft, pulse.ifft, pulse.save_field, 
                      plasma.save_plasma_density, plasma.atom, 
                      plasma.load_num_den, plasma.load_plasma_den,
                      pulse.threads, temp, n2, ionization))


def pulse_multispecies(pulse, multi):
    """ Propagates a pulse through a gas, ionizing and refracting as it goes.
    
    Parameters
    ----------
    pulse : Pulse class
        The laser pulse to propagate through the plasma.
    plasma : Plasma class
        The gas to propagate the laser pulse through.
    """
    pulse.e = pcalc.multispecies_refraction(pulse.e, pulse.x, pulse.y,
                      multi.z, pulse.t, pulse.lam, pulse.z[-1], 
                      pulse.save_field, multi.save_density, multi.load_den)


def beam_phase(beam, phase):
    """ Applies a phase mask to a optical beam, either a pulse or laser.
    
    Parameters
    ----------
    beam : Pulse or Laser class
        The optical beam the apply the phase mask to.
    phase : Phase class
        The phase mask to apply to the beam.
    """
    beam.set_field(beam.e * np.exp(1j*phase.phi))
    
    
def beam_intensity(beam, intensity):
    """ Applies a transmission mask to a optical beam, either a pulse or laser.
    
    Parameters
    ----------
    beam : Pulse or Laser class
        The optical beam the apply the phase mask to.
    intensity : Intensity class
        The intensity mask to apply to the beam.
    """
    beam.set_field(beam.e * intensity.t)


def beam_plasma(beam, plasma):
    """ Propagates a weak beam through a plasma without an ionization.
    
    Parameters
    ----------
    beam : Laser class
        The optical beam to propagate through the plasma.
    plasma : Plasma class
        The plasma to propagate the laser pulse through.
    """
    nh = 1.0 + plasma.n0*plasma.atom['alpha']*5.0e-8
    def loadnh(ind):
        return nh
    def loadn(ind):
        ne = plasma.load_plasma_den(ind)
        nplasma = -beam.lam**2 * 4.47869e-5
        ngas = plasma.atom['alpha']*5.0e-8
        dn = nplasma - ngas
        return dn * ne
    beam.e = lcalc.beam_prop(beam.e, beam.x, beam.y, plasma.z, beam.lam, loadnh,
                             beam.z[-1], beam.fft, beam.ifft, beam.save_field,
                             loadn)
    
    
def beam_index(beam, index):
    """ Propagate a beam through a region with varying index of refraction. 
    
    Parameters
    ----------
    beam : Laser class
        The optical beam to propagate through the region.
    index : Index class
        The index of refraction object.
    nh : double
        The background index of refraction. 
    """
    beam.e = lcalc.beam_prop(beam.e, beam.x, beam.y, index.z, beam.lam, index.loadnh,
                             beam.z[-1], beam.fft, beam.ifft, beam.save_field,
                             index.loadn)


def electron_plasma(electron, plasma, z, dumpPeriod, n):
    """ Propagate an electron beam through an ion column. 
    
    Parameters
    ----------
    electron : ElectronBeam class
        The electron beam to propagate through the plasma.
    plasma : Plasma class
        The plasma that the electron beam will be propagating through.
    z : array-like
        The spatial grid used to set the step size for the electron beam.
    dumpPeriod : int
        How frequently to save the electron beam to disk.
    n : int
        Number of threads to run on.
    """
    electron.ptcls = ecalc.electron_propagation_plasma(electron.ptcls,
                            z*1e-6, 0.0, plasma.get_ne(z), dumpPeriod,
                            electron.save_ptcls, plasma.dgammadz, n)
