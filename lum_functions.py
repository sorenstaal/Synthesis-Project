#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:30:25 2023

@author: sorenstaal
"""
# Preliminary
import numpy as np
import scipy.integrate as integrate
from scipy.stats import truncnorm

import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

os.environ["PATH"] = '/Users/sorenstaal/opt/anaconda3/bin:/Users/sorenstaal/opt/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin'
os.chdir('/Users/sorenstaal/Documents/Uni/Kandidat/3. semester/Synthesis Project/Scripts')


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'text.latex.preamble': r'\usepackage{amsmath}'})


# utility functions

# quick functions for converting between IR luminosity
#  and HCN line luminosity

def LHCNtoLIR(L_HCN, a = 1, b = 3):
    return L_HCN ** a * 10 ** b

def LIRtoLHCN(L_IR, a = 1, b = 3):
    return L_IR ** (1 / a) * 10 ** (-b / a)

def get_Phistar(z, z_turn = 2.0, z_w = 2.0, Phi0 = 10**(-3.5), 
                psi1 = 0, psi2 = -6.5):
    """
    Characteristic number density as function of redshift. Default values and
    model from Casey et al. 2018a and Zavala et al. 2021

    Parameters
    ----------
    z : scalar or array
        input redshift.
    z_turn : scalar, optional
        turn-over redshift. The default is 2.0.
    z_w : scalar, optional
        width of turnover. The default is 2.0.
    Phi0 : scalar, optional
        number density at z = 0. The default is 10**(-3.5) [Mpc^-3 dex^-1].
    psi1 : scalar, optional
        redshift evolution at z << z_turn. The default is 0.
    psi2 : scalar, optional
        redshift evolution at z >> z_turn. The default is -6.5.

    Returns
    -------
    Phistar, characteristic number density at given redshift [Mpc^-3 dex^-1].
    Same type as input redshift

    """
    # parameterize
    x = np.log10(1 + z)
    xt = np.log10(1 + z_turn)
    xw = z_w / (np.log(10) * (1 + z_turn))
    
    log_Phi = ((psi2 - psi1) * xw / (2 * np.pi)) * \
        (np.log(np.cosh(np.pi * (x - xt) / xw)) - \
         np.log(np.cosh( - np.pi * xt / xw))) + \
        (psi2 - psi1) / 2 * x + np.log10(Phi0)
    Phistar = 10**(log_Phi) * (1 + z) ** (psi1)
    
    return Phistar



def get_Lstar(z, z_turn = 2.0, z_w = 2.0, L0 = 10 ** (11.1),
              g1 = 2.8, g2 = 1):
    """
    Evolution of the knee of the luminosity function with redshift.
    Default values and model from Casey et al. 2018a and 
    Zavala et al. 2021

    Parameters
    ----------
    z : scalar or array
        input redshift.
    z_turn : scalar, optional
        turn-over redshift. The default is 2.0.
    z_w : scalar, optional
        width of turnover. The default is 2.0.
    L0 : scalar, optional
        Knee of the IR luminosity function at z = 0.
        The default is 10 ** (11.1) [L_Sun].
    g1 : scalar, optional
        Redshift evolution at z << z_turn. The default is 2.8.
    g2 : scalar, optional
        redshift evolution at z >> z_turn. The default is 1.0.

    Returns
    -------
    Knee of the IR luminosity function [L_Sun].
    Same type as input redshift

    """
    # parameterize
    x = np.log10(1 + z)
    xt = np.log10(1 + z_turn)
    xw = z_w / (np.log(10) * (1 + z_turn))
    
    log_L = ((g2 - g1) * xw / (2 * np.pi)) * \
        (np.log(np.cosh(np.pi * (x - xt) / xw)) - \
         np.log(np.cosh( - np.pi * xt / xw))) + \
        ((g2 - g1) / 2) * x + np.log10(L0)
    
    Lstar = 10 ** log_L * (1 + z) ** g1
    
    return Lstar




def get_HCNLstar(z, z_turn = 2.0, z_w = 2.0, L0 = 10 ** (11.1),
              g1 = 2.8, g2 = 1.0, a = 1, b = 3):
    """
    Evolution of the knee of the HCN line luminosity function with redshift.
    Default values and model from Casey et al. 2018a and 
    Zavala et al. 2021
    a and b conversion factors from Rybak et al. 2022

    Parameters
    ----------
    z : scalar or array
        input redshift.
    z_turn : scalar, optional
        turn-over redshift. The default is 2.0.
    z_w : scalar, optional
        width of turnover. The default is 2.0.
    L0 : scalar, optional
        Knee of the IR luminosity function at z = 0.
        The default is 10 ** (11.1) [L_Sun].
    g1 : scalar, optional
        Redshift evolution at z << z_turn. The default is 2.8.
    g2 : scalar, optional
        redshift evolution at z >> z_turn. The default is 1.0.
    a :  scalar
    HCN-IR conversion factor. The default is 1.
    b : scalar
    HCN-IR conversion factor. The default is 3.

    Returns
    -------
    Knee of the HCN line luminosity function [L_Sun].
    Same type as input redshift
    
    """
    L0_HCN = LIRtoLHCN(L0, a, b)
    
    x = np.log10(1 + z)
    xt = np.log10(1 + z_turn)
    xw = z_w / (np.log(10) * (1 + z_turn))
    
    f = ((g2 - g1) * xw / (2 * np.pi)) * \
        (np.log(np.cosh(np.pi * (x - xt) / xw)) - \
         np.log(np.cosh( - np.pi * xt / xw))) + \
        ((g2 - g1) / 2) * x
    
    log_L = f / a + np.log10(L0_HCN)
    
    Lstar_HCN = 10 ** log_L * (1 + z) ** g1
    
    return Lstar_HCN 

    


def get_IRlumfunc(z, L, a_LF = -0.42, b_LF = -3.0):
    """
    IR luminosity function from input luminosity and redshift.
    Default values and model from Casey et al. 2018a and Zavala et al. 2021

    Parameters
    ----------
    z : scalar
        input redshift.
    L : 1D array
        input luminosity [L_Sun].
    a_LF : scalar, optional
        Faint-end slope of the IR luminosity function. The default is -0.42.
    b_LF : scalar, optional
        Brigbht-end slope of the IR luminosity function. The default is -3.0.

    Returns
    -------
    IRlumfunc (1D array), IR luminosity density [Mpc^-3 dex^-1].


    """
    Lstar = get_Lstar(z)
    Phistar = get_Phistar(z)
    
    IR_lum = np.zeros(len(L))
    
    IR_lum[L < Lstar] = Phistar * (L[L < Lstar] / Lstar) ** a_LF
    IR_lum[L >= Lstar] = Phistar * (L[L >= Lstar] / Lstar) ** b_LF
    
    return IR_lum


def get_HCNlumfunc(z, L, a_LF = -0.42, b_LF = -3.0, a = 1, b = 3):
    """
    IR luminosity function from input luminosity and redshift.
    Default values and model from Casey et al. 2018a and Zavala et al. 2021
    a and b conversion factors from Rybak et al. 2022

    Parameters
    ----------
    z : scalar
        input redshift.
    L : 1D array
        input HCN line luminosity [K km s^-1 pc^2].
    a_LF : scalar, optional
        Faint-end slope of the IR luminosity function. The default is -0.42.
    b_LF : scalar, optional
        Brigbht-end slope of the IR luminosity function. The default is -3.0.

    Returns
    -------
    HCNlumfunc (1D array), HCN luminosity density [Mpc^-3 dex^-1].


    """
    Lstar = get_HCNLstar(z, a = a, b = b)
    Phistar = get_Phistar(z)
    
    HCN_lum = np.zeros(len(L))
    
    HCN_lum[L < Lstar] = Phistar * (L[L < Lstar] / Lstar) ** (a * a_LF)
    HCN_lum[L >= Lstar] = Phistar * (L[L >= Lstar] / Lstar) ** (a * b_LF)

    return HCN_lum


def IR_density(z, a_LF = -0.42, b_LF = -3.0):
    """
    Integrated IR luminsoty density, from the integrated IR luminosity function 
    [L_Sun Mpc^-3]. Default values from Zavala et al. 2021

    Parameters
    ----------
    z : scalar or 1D array
        input redshift.
    a_LF : scalar, optional
        Faint-end slope of the luminosity function. The default is -0.42.
    b_LF : scalar, optional
        Bright-end slope of the luminosity function. The default is -3.0.


    Returns
    -------
    Integrated IR luminsoity density [L_Sun Mpc^-3], same type as input
    redshift.
    
    """
    
    Lstar = get_Lstar(z)
    Phistar = get_Phistar(z)
    rho_IR = Phistar * Lstar * (1 / (a_LF + 1) - 1 / (b_LF + 1))
    
    return rho_IR
    




def HCN_density(z, a_LF = -0.42, b_LF = -3.0, a = 1, b = 3):
    """
    Integrated HCN line density, from the integrated HCN luminosity function 
    [K km s^-1 pc^2 Mpc^-3]. Default values from Zavala et al. 2021 and
    Rybak et al. 2022

    Parameters
    ----------
    z : scalar or 1D array
        input redshift.
    a_LF : scalar, optional
        Faint-end slope of the luminosity function. The default is -0.42.
    b_LF : scalar, optional
        Bright-end slope of the luminosity function. The default is -3.0.
    a : scalar, optional
        HCN-IR conversion factor. The default is 1.
    b : scalar, optional
        HCN-IR conversion factor. The default is 3.

    Returns
    -------
    Integrated HCN line density [K km s^-1 pc^2 Mpc^-3], same type as input
    redshift.

    """
    Lstar = get_HCNLstar(z, a = a, b = b)
    Phistar = get_Phistar(z)
    rho_HCN = Phistar * Lstar * (1 / (a * a_LF + 1) - 1 / (a * b_LF + 1))
    
    return rho_HCN



def HCN_density_num(z, a_LF = -0.42, b_LF = -3.0, a = 1,
                    L_low = 0, L_high = np.inf):
    """
    Numerically integrated HCN line density, 
    from the integrated HCN luminosity function 
    [K km s^-1 pc^2 Mpc^-3]. Default values from Zavala et al. 2021 and
    Rybak et al. 2022

    Parameters
    ----------
    z : scalar or 1D array
        input redshift.
    a_LF : scalar, optional
        Faint-end slope of the luminosity function. The default is -0.42.
    b_LF : scalar, optional
        Bright-end slope of the luminosity function. The default is -3.0.
    a : scalar, optional
        HCN-IR conversion factor. The default is 1.
    L_low : scalar, optional
        Lower integration limit. The default is 0.
    L_high : scalar, optional
        Higher integration limit. The default is inf.

    Returns
    -------
    Numerically integrated HCN line density [K km s^-1 pc^2 Mpc^-3], same type as input
    redshift.

    """
    rho_HCN_num = np.zeros(len(z))

    for i in range(len(z)):
        
        Phistar = get_Phistar(z[i])
        Lstar = get_HCNLstar(z[i])
        
        L1 = np.logspace(0, np.log10(Lstar), 1000)
        L2 = np.logspace(np.log10(Lstar), 50, 1000)
        
        Phi_HCN_lower = Phistar * Lstar ** (-a * a_LF) * L1 ** (a * a_LF)
        Phi_HCN_upper = Phistar * Lstar ** (-a * b_LF) * L2 ** (a * b_LF)
        

    
    
        # integrate
        int1 = integrate.simpson(Phi_HCN_lower, L1)
        int2 = integrate.simpson(Phi_HCN_upper, L2)

    
        rho_HCN_num[i] = int1 + int2
        
    return rho_HCN_num

def H2_density(z, alpha1 = 3.1, alpha2 = 0.8, L_cutoff = 10**11,
               a_LF = -0.42, b_LF = -3.0, a = 1.12, b = 0.5):
   
    L_cutoff = LIRtoLHCN(L_cutoff, a = a, b = b)
    
    # Lstar = get_HCNLstar(z, a = a, b = b)
    # Phistar = get_Phistar(z)
    
    M_H2 = np.zeros(len(z))
    
    for i in range(len(z)):
        
        Phistar = get_Phistar(z[i])
        Lstar = get_HCNLstar(z[i], a = a, b = b)
        
        if Lstar > L_cutoff:
            L1 = np.logspace(0, np.log10(L_cutoff), 1000)
            L2 = np.logspace(np.log10(L_cutoff), np.log10(Lstar), 1000)
            L3 = np.logspace(np.log10(Lstar), 100, 1000)
            
            Phi_CO_1 = Phistar * Lstar ** (-a * a_LF) * L1 ** (a * a_LF)
            Phi_CO_2 = Phistar * Lstar ** (-a * a_LF) * L2 ** (a * a_LF)
            Phi_CO_3 = Phistar * Lstar ** (-a * b_LF) * L3 ** (a * b_LF)
            
            int1 = integrate.simpson(Phi_CO_1, L1)
            int2 = integrate.simpson(Phi_CO_2, L2)
            int3 = integrate.simpson(Phi_CO_3, L3)
            
            M_H2[i] = alpha1 * (int1) + alpha2 * (int2 + int3)
            
        else:
            L1 = np.logspace(0, np.log10(Lstar), 1000)
            L2 = np.logspace(np.log10(Lstar), np.log10(L_cutoff), 1000)
            L3 = np.logspace(np.log10(L_cutoff), 100, 1000)
            
            Phi_CO_1 = Phistar * Lstar ** (-a * a_LF) * L1 ** (a * a_LF)
            Phi_CO_2 = Phistar * Lstar ** (-a * b_LF) * L2 ** (a * b_LF)
            Phi_CO_3 = Phistar * Lstar ** (-a * b_LF) * L3 ** (a * b_LF)
            
            int1 = integrate.simpson(Phi_CO_1, L1)
            int2 = integrate.simpson(Phi_CO_2, L2)
            int3 = integrate.simpson(Phi_CO_3, L3)
            
            M_H2[i] = alpha1 * (int1 + int2) + alpha2 * (int3)
            
    
    
    return M_H2


def sample_conversions(z, a_sd, b_sd, end_dir, fn, a = 1, b = 3, num = 1000):
    """
    Simple function for generating HCN density range plots based on a and b 
    conversion factor standard deviations. Assumes a and b follow
    Gaussian distrubtions and are non-correlated

    Parameters
    ----------
    z : scalar or 1D array
        input redshift.
    a_sd : scalar
        standard deviation on conversion factor a.
    b_sd : scalar
        standard deviation on conversion factor b.
    end_dir : string
        directory to save plot in.
    fn : string
        filename
    a : scalar, optional
        IR-HCN conversion factor. The default is 1.
    b : scalar, optional
        IR-HCN conversion factor. The default is 3.
    num : int, optional
        number of (Gaussian) samples. The default is 1000.

    Returns
    -------
    HCN density plot, with shaded region from samples

    """
    
    rho_HCN_main = HCN_density(z)
    
    a_sample = np.random.normal(a, a_sd, num)
    b_sample = np.random.normal(b, b_sd, num)
    
    print("min a_sample: {}".format(np.min(a_sample)))
    print("min b_sample: {}".format(np.min(b_sample)))
    
    fig, axes = plt.subplots(figsize = (7,5))
    
    #axes.plot(z, rho_HCN_main, '-k', lw = 2)
    
    for a, b in zip(a_sample, b_sample):
        rho_HCN_sample = HCN_density(z, a = a, b = b)
        axes.plot(z, rho_HCN_sample, c = 'tab:gray', lw = 2, alpha = 0.5)
    axes.plot(z, rho_HCN_main, '-k', lw = 2, alpha = 1)
    
    axes.set_yscale('log')
    axes.set_xlabel(r'Redshift', fontsize = 20)
    axes.set_ylabel(r"$\rho_{\text{HCN}}$ [K km s$^{-1}$ pc$^2$ Mpc$^{-3}$]", fontsize = 20)
    
    
    axes.tick_params(labelsize = 16, direction = 'in')
    axes.grid()
    
    plt.tight_layout()
    plt.savefig(os.path.join(end_dir, fn))
    plt.clf()
    plt.close()




def HCN_density_sigma(z, a_sd, b_sd, a = 1, b = 3, end_dir = False, fn = False,
                      plot = False):
    """
    Simple function for making rho_HCN plot with 1 sigma error
    shaded region

    Parameters
    ----------
    z : 1D array
        input redshift.
    a_sd : scalar
        standard deviation of HCN-IR conversion factor a.
    b_sd : scalar
        standard deviation of HCN-IR conversion factor a.
    end_dir : string
        directory to save plot in.
    fn : string
        filename (including filetype.
    a : scalar, optional
        HCN-IR conversion factor a. The default is 1.
    b : scalar, optional
        HCN-IR conversion factor b. The default is 3.
    num : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    Plot of rho_HCN with +/- 1 sigma shaded region along with mean
    rho_HCN, rho_HCN_lower and rho_HCN_upper

    """
    
    a_lower = a - a_sd
    a_upper = a + a_sd
    
    b_lower = b - b_sd
    b_upper = b + b_sd
    
    rho_HCN_main = HCN_density(z)
    rho_HCN_lower = HCN_density(z, a = a_lower, b = b_lower)
    rho_HCN_upper = HCN_density(z, a = a_upper, b = b_upper)
    
    if plot:
        fig, axes = plt.subplots(figsize = (7,5))
        
        axes.fill_between(z, rho_HCN_lower, rho_HCN_upper, color = 'tab:gray',
                          alpha = 0.5, lw = 0)
        axes.plot(z, rho_HCN_main, '-k', lw = 2, alpha = 1)
        
        axes.set_yscale('log')
        axes.set_xlabel(r'Redshift', fontsize = 20)
        axes.set_ylabel(r"$\rho_{\text{HCN}}$ [K km s$^{-1}$ pc$^2$ Mpc$^{-3}$]", fontsize = 20)
        
        
        axes.tick_params(labelsize = 16, direction = 'in')
        axes.grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(end_dir, fn))
        plt.clf()
        plt.close()
    return rho_HCN_main, rho_HCN_lower, rho_HCN_upper

def get_Mgas(z, a_sd, b_sd, alpha_lower = 13, alpha_upper = 89, a = 1.0,
              b = 3.0, alpha = 32, plot = False, fn = False, end_dir = False):
    
    a_lower = a - a_sd
    a_upper = a + a_sd
    
    b_lower = b - b_sd
    b_upper = b + b_sd
    
    rho_HCN_main = HCN_density(z, a = a, b = b)
    rho_HCN_lower = HCN_density(z, a = a_lower, b = b_lower)
    rho_HCN_upper = HCN_density(z, a = a_upper, b = b_upper)
    
    M_dense = rho_HCN_main * alpha
    M_dense_lower = rho_HCN_lower * alpha_lower
    M_dense_upper = rho_HCN_upper * alpha_upper
    
    if plot:
        fig, axes = plt.subplots(figsize = (7,5))
        
        axes.fill_between(z, M_dense_lower, M_dense_upper, color = 'tab:gray',
                          alpha = 0.5, lw = 0)
        axes.plot(z, M_dense, '-k', lw = 2, alpha = 1)
        
        axes.set_yscale('log')
        axes.set_xlabel(r'Redshift', fontsize = 20)
        axes.set_ylabel(r"M$_{\text{dense}}$ [M$_\odot$]", fontsize = 20)
        
        
        axes.tick_params(labelsize = 16, direction = 'in')
        axes.grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(end_dir, fn))
        plt.clf()
        plt.close()
    
    return M_dense, M_dense_lower, M_dense_upper

def powerlaw(z, A, B, C, D):
    """
    Powerlaw function for getting mass density as function of redshift (z)
    with parameters A, B, C and D. From Walter et al. 2020
    The next functions are the derivatives of the powerlaw, for error
    propagation.
    """
    return A * (1 + z) ** B / ((1 + ((1 + z) / C) ** D))

def dpl_dA(z, A, B, C, D):
    return (1 + z) ** B / (1 + ((1 + z) / C)) ** D

def dpl_dB(z, A, B, C, D):
    return powerlaw(z, A, B, C, D) * np.log(1 + z)

def dpl_dC(z, A, B, C, D):
    return powerlaw(z, A, B, C, D) * D * ((1 + z) / C) ** D / \
        ((1 + ((1 + z) / C) ** D) * C)

def dpl_dD(z, A, B, C, D):
    return powerlaw(z, A, B, C, D) * ((1 + z) / C) ** D * (-np.log((1 + z) / C)) \
        / ((1 + ((1 + z) / C) ** D))

def hyp_tang(z, A, B, C):
    """
    Hyperbolic tangent function for getting HI mass density as function of 
    redshift (z) with parameters A, B, C. From Walter et al. 2020
    The next functions are the derivatives for error propagation
    """
    return A * np.tanh(1 + z - B) + C

def dht_dA(z, A, B, C):
    return hyp_tang(z, A, B, C) / A

def dht_dB(z, A, B, C):
    return -A * (1 + np.tanh(1 + z - B) ** 2)


def fit_powerlaw(z, InPar, alpha_lower = 13, alpha_upper = 89,
                 alpha = 32):
    """
    Function for fitting powerlaw from Walter et al. 2020 to mass density
    function. In case of asymmetric lower and upper bounds on alpha conversion
    factor, the mean is used, as scipy's curve_fit function only allows
    symmetric errors. Default alpha values from Tunnard et al. 2021

    Parameters
    ----------
    z : scalar or 1D array
        input array.
    InPar : list or array of length 4
        initial parameters for A, B, C and D. Note that curve_fit is highly
        susceptible to these and that parameter A is also somewhat
        normalized in the function
    alpha_lower : scalar, optional
        lower bound on alpha conversion factor. The default is 13.
    alpha_upper : scalar, optional
        upper bound on alpha conversion factor. The default is 89.
    alpha : scalar, optional
        alpha conversion factor. The default is 32.

    Returns
    -------
    M_dense_pl : scalar or 1D array (same type as input redshift)
        Gas mass density [M_Sun Mpc^-3].
    M_dense_pl_sd : scalar or 1D array (same type as input redshift)
        Gas mass density errors.
    cov : 4 x 4 array
        covariance matrix of the four parameters.
    params : list or array of length 4
        Output parameters.

    """
    
    M_dense = HCN_density(z) * alpha
    
    InPar[0] /= 10 ** 7
    
    sd = (alpha_lower + alpha_upper ) / 2 - alpha * HCN_density(z)
    
    params, cov = curve_fit(powerlaw, z, M_dense / 10 ** 7, p0 = InPar, 
                             sigma = sd / (10 ** 7), maxfev = 5000)
    sigma = np.sqrt(np.diagonal(cov))
    sigma[0] *= 10 ** 7
    params[0] *= 10 ** 7
    
    M_dense_pl = powerlaw(z, *params)
    
    A_sd = dpl_dA(z, *params) * sigma[0]
    B_sd = dpl_dB(z, *params) * sigma[1]
    C_sd = dpl_dC(z, *params) * sigma[2]
    D_sd = dpl_dC(z, *params) * sigma[3]
    
    M_dense_pl_sd = np.sqrt(A_sd ** 2 + B_sd ** 2 + C_sd ** 2 + D_sd ** 2)
    
    
    return  M_dense_pl, M_dense_pl_sd, cov, params

def CO_density(z, a_LF = -0.42, b_LF = -3.0, a = 0.99, b = 1.9):
    """
    Quick function for getting CO luminosity density as function of redshift.
    Same form as function for HCN density, as same relation with IR
    luminosity is assumed, just with different devault values.
    Slope (a_LF and b_LF) devault values from Zavala et al. 2021 and
    IR-CO values (a and b) from Greve et al. 2014

    Parameters
    ----------
    z : scalar or 1D array
        inout redshift.
    a_LF : scalar, optional
        faint-end slope of the luminosity function. The default is -0.42.
    b_LF : scalar, optional
        bright-end slope of the luminosity function. The default is -3.0.
    a : scalar, optional
        CO-IR conversion factor. The default is 0.99.
    b : scalar, optional
        CO-IR conversion factor. The default is 1.9.

    Returns
    -------
    rho_CO : scalar or 1D array (same type as input redshift)
        CO luminosity density [K km/s pc^2 Mpc^-3].

    """
    
    
    Lstar = get_HCNLstar(z, a = a, b = b)
    Phistar = get_Phistar(z)
    
    rho_CO = Phistar * Lstar * (1 / (a * a_LF + 1) - 1 / (a * b_LF + 1))
    
    return rho_CO

def get_d_lum(z, H0 = 67.8):
    
    c = 299792.458  #[km/s]
    
    d_lum = 2 * c / H0 * (1 - np.sqrt(1 / (1 + z))) * (1 + z)
    
    return d_lum

def lum_to_intensity(z, L, H0 = 67.8):
    
    d_lum = get_d_lum(z, H0)
    
    I = L / (4 * np.pi * d_lum ** 2)
    
    return I

def param_conversion(z, L, Z, H0 = 67.8):
    
    I = lum_to_intensity(z, L, H0)
    
    a_CO = 10.7 * I ** (-0.32)
    a_CO[a_CO > 6.3] = 6.3
    
    a_CO /= (Z ** (0.65))
    
    return a_CO

def get_depletion_time(z, M_gas, A = 0.0158, A_err = 0.0010, B = 2.88,
                       B_err = 0.16, C = 2.75, C_err = 0.11, D = 5.88,
                       D_err = 0.15):
    
    cSFRd = powerlaw(z, A, B, C, D)
    
    A_sd = dpl_dA(z, A, B, C, D) * A_err
    B_sd = dpl_dB(z, A, B, C, D) * B_err
    C_sd = dpl_dC(z, A, B, C, D) * C_err
    D_sd = dpl_dD(z, A, B, C, D) * D_err
    
    tau_depl = M_gas / cSFRd
    tau_depl_err = np.sqrt(A_sd ** 2 + B_sd ** 2 + C_sd ** 2 + D_sd ** 2) \
        * tau_depl
    
    return tau_depl, tau_depl_err
    










if __name__ == "__main__":
    # InPar = [10**7, 3, 2.3, 5.1]
    z_array = np.linspace(0, 10, 1000)
    
    # M_dense_pl, err, cov, params = fit_powerlaw(z_array, InPar)
    
    # fig, axes = plt.subplots(figsize = (7,5))
    
    # axes.fill_between(z_array, M_dense_pl - err, M_dense_pl + err,
    #                   color = 'b', alpha = 0.5, lw = 0.3)
    # axes.plot(z_array, M_dense_pl, 'b', lw = 3, alpha = 1, label = r'Powerlaw')
    
    # a = 1
    # b = 3
    # a_sd = 0.1 * a
    # b_sd = 0.1 * b
    
    # M_dense, M_dense_lower, M_dense_upper = get_Mgas(z_array, a_sd = a_sd, b_sd = b_sd)
    # axes.fill_between(z_array, M_dense_lower, M_dense_upper, color = 'tab:gray',
    #                   alpha = 0.5, lw = 0)
    # axes.plot(z_array, M_dense, '-k', lw = 2, alpha = 1, 
    #           label = r'Main model')
    
    # axes.set_yscale('log')
    
    
    # axes.set_yscale('log')
    # axes.set_xlabel(r'Redshift', fontsize = 20)
    # axes.set_ylabel(r"$\rho_{\text{dense}}$ [M$_\odot$ Mpc$^{-3}$]", fontsize = 20)
    
    

    # axes.tick_params(labelsize = 16, direction = 'in')
    # axes.legend(prop = {'size': 14}, frameon = False)
    # axes.grid()
    
    # plt.tight_layout()
    # plt.savefig('../Figures/rho_dense_pl.pdf')
    # plt.clf()
    # plt.close()
    
    # os.environ["PATH"] = '/Users/sorenstaal/opt/anaconda3/bin:/Users/sorenstaal/opt/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin'

    # os.chdir('/Users/sorenstaal/Documents/Uni/Kandidat/3. semester/Synthesis Project/Scripts')
    
    # end_dir = '../Figures'
    # file_name = 'measured_v_model2.pdf'
    # z_array = np.linspace(0, 10, 1000)
    
    # # based on 30% standard deviation
    
    # a = 1
    # b = 3
    # a_sd = 0.1 * a
    # b_sd = 0.1 * b
    
    # M_dense, M_dense_lower, M_dense_upper = get_Mgas(z_array, 
    #                                                  a_sd = a_sd, 
    #                                                  b_sd = b_sd,
    #                                                  alpha_lower = 10,
    #                                                  alpha_upper = 10,
    #                                                  alpha = 10)
   
    # rho_HI = np.loadtxt('../Data/rho_HI.txt')
    # rho_H2 = np.loadtxt('../Data/rho_H2.txt')
    
    # fig, axes = plt.subplots(figsize = (7,5))
    
    # axes.fill_between(z_array, M_dense_lower, M_dense_upper, color = 'tab:gray',
    #                   alpha = 0.5, lw = 0, label = r'$\rho_{\text{dense}}$ range')
    # axes.plot(z_array, M_dense, '-k', lw = 2, alpha = 1, 
    #           label = r'$\rho_{\text{dense}}$')
    
    # axes.errorbar(rho_H2[:,1], rho_H2[:,3] * 10 ** 8, xerr = [rho_H2[:,1] - rho_H2[:,0], rho_H2[:,2] - rho_H2[:,1]], yerr = 
    #               [rho_H2[:,4] * 10 ** 8, rho_H2[:,5] * 10 ** 8], 
    #               label = r'$\rho_{\text{H2}}$', fmt = 'bo', alpha = 0.7, 
    #               elinewidth = 0.5, markersize = 4)

    # axes.errorbar(rho_HI[:,1], rho_HI[:,3] * 10 ** 8, xerr = [rho_HI[:,1] - rho_HI[:,0], rho_HI[:,2] - rho_HI[:,1]], yerr = 
    #               [rho_HI[:,4] * 10 ** 8, rho_HI[:,5] * 10 ** 8], 
    #               label = r'$\rho_{\text{HI}}$', fmt = 'ro', alpha = 0.7, 
    #               elinewidth = 0.5, markersize = 4)
    
    # # axes.set_xticks(np.arange(0, 11))
    # # axes.set_xscale('symlog')
    
    # axes.set_yscale('log')
    # axes.set_xlabel(r'Redshift', fontsize = 20)
    # axes.set_ylabel(r"$\rho_{\text{gas}}$ [M$_\odot$ Mpc$^{-3}$]", fontsize = 20)
    
    
    # axes.legend(prop = {'size': 14}, frameon = False)
    # axes.tick_params(labelsize = 16, direction = 'in')
    # axes.grid()
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(end_dir, file_name))
    # plt.clf()
    # plt.close()    


