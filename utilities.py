#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:25:04 2023

@author: sorenstaal
"""
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np
import lum_functions as lf

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QSizePolicy
)

from PyQt5 import QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pandas as pd

data_dir = 'Data'




class MplCanvas(FigureCanvas):
    """ Creates a matplotlib FigureCanvas class for use in other plots """
    def __init__(self, parent = None, width = 10, height = 7, dpi = 80):
        fig = Figure(figsize = (width, height), dpi = dpi)

        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes.plot()
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class DummyPlot(MplCanvas):
    """ Dummy plot for testing functions """
    def make_figure(self):
        
        self.axes.cla()
        
        x = np.linspace(0, 2)
        y = np.sin(x * np.pi)
        
        self.axes.plot(x, y)
        self.axes.grid()
        
        self.draw()
        
    
        
class MdensePlot(MplCanvas):
    """ Try plotting Mgas with class """
    def make_figure(self, alpha, a, b, a_sd, b_sd, use_data = False):
        if alpha == "":
            alpha = 0

        self.axes.cla()
        
        alpha = float(alpha)
        a = float(a)
        b = float(b)
        a_sd = float(a_sd)
        b_sd = float(b_sd)
        
        a_sd *= a / 100
        b_sd *= b / 100
        
        z_array = np.linspace(0, 10, 1000)
        
        M_dense = lf.HCN_density(z_array, a = a, b = b) * alpha
        
        M_dense_lower = lf.HCN_density(z_array, a = a - a_sd, b = b - b_sd) * alpha
        M_dense_upper = lf.HCN_density(z_array, a = a + a_sd, b = b + b_sd) * alpha
        
        self.axes.plot(z_array, M_dense, lw = 2,
                       label = r'Modelled $\rho_{\text{dense}}$')
        
        if a_sd > 0 or b_sd > 0:
            self.axes.fill_between(z_array, M_dense_lower, M_dense_upper,
                                   color = 'b', alpha = 0.3, lw = 0)
        
        if use_data:
            rho_HI_fn = 'rho_HI.txt'
            rho_H2_fn = 'rho_H2.txt'
            
            rho_HI = np.loadtxt(os.path.join(data_dir, rho_HI_fn))
            rho_H2 = np.loadtxt(os.path.join(data_dir, rho_H2_fn))
            
            self.axes.errorbar(rho_H2[:,1], rho_H2[:,3] * 10 ** 8, xerr = [rho_H2[:,1] - rho_H2[:,0], rho_H2[:,2] - rho_H2[:,1]], yerr = 
                          [rho_H2[:,4] * 10 ** 8, rho_H2[:,5] * 10 ** 8], 
                          label = r'$\rho_{\text{H2}}$', fmt = 'bo', alpha = 0.7, 
                          elinewidth = 0.5, markersize = 4)

            self.axes.errorbar(rho_HI[:,1], rho_HI[:,3] * 10 ** 8, xerr = [rho_HI[:,1] - rho_HI[:,0], rho_HI[:,2] - rho_HI[:,1]], yerr = 
                          [rho_HI[:,4] * 10 ** 8, rho_HI[:,5] * 10 ** 8], 
                          label = r'$\rho_{\text{HI}}$', fmt = 'ro', alpha = 0.7, 
                          elinewidth = 0.5, markersize = 4)
            self.axes.legend(prop = {'size': 14}, frameon = False)
            
        
        self.axes.set_xlabel(r'Redshift', fontsize = 20)
        self.axes.set_ylabel(r'$\rho_{\text{dense}}$ [M$_\odot$ Mpc$^{-3}$]', fontsize = 20)
        self.axes.set_yscale('log')
        
        self.axes.grid()
        self.axes.tick_params(labelsize = 16, direction = 'in')
        
        
        self.draw()

class MH2Plot(MplCanvas):
    def make_figure(self, alpha1, alpha2, a, b, a_sd, b_sd,
                    use_data = False, use_walter = False):
        if alpha1 == "":
            alpha1 = 0
        if alpha2 == "":
            alpha2 = 0
        self.axes.cla()
        
        alpha1 = float(alpha1)
        alpha2 = float(alpha2)
        a = float(a)
        b = float(b)
        a_sd = float(a_sd)
        b_sd = float(b_sd)
        
        # convert from percent
        a_sd = a_sd * a / 100
        b_sd = b_sd * b / 100
        
        z_array = np.linspace(0, 10, 1000)
        
        M_H2 = lf.H2_density(z_array, a = a, b = b, alpha1 = alpha1,
                             alpha2 = alpha2)
        
        M_H2_lower = lf.H2_density(z_array, a = a - a_sd, b = b - b_sd, 
                                   alpha1 = alpha1, alpha2 = alpha2)
        M_H2_upper = lf.H2_density(z_array, a = a + a_sd, b = b + b_sd, 
                                   alpha1 = alpha1, alpha2 = alpha2)
        
        
        self.axes.plot(z_array, M_H2, '-k', lw = 2,
                       label = r'Modelled $\rho_{\text{H2}}$')
        
        if b_sd > 0 or a_sd > 0:
            self.axes.fill_between(z_array, M_H2_lower, M_H2_upper,
                                   color = 'tab:gray', alpha = 0.5,
                                   lw = 0)
        
        if use_data:
            rho_HI_fn = 'rho_HI.txt'
            rho_H2_fn = 'rho_H2.txt'
            
            rho_HI = np.loadtxt(os.path.join(data_dir, rho_HI_fn))
            rho_H2 = np.loadtxt(os.path.join(data_dir, rho_H2_fn))
            
            self.axes.errorbar(rho_H2[:,1], rho_H2[:,3] * 10 ** 8, xerr = [rho_H2[:,1] - rho_H2[:,0], rho_H2[:,2] - rho_H2[:,1]], yerr = 
                          [rho_H2[:,4] * 10 ** 8, rho_H2[:,5] * 10 ** 8], 
                          label = r'$\rho_{\text{H2}}$', fmt = 'bo', alpha = 0.7, 
                          elinewidth = 0.5, markersize = 4)

            self.axes.errorbar(rho_HI[:,1], rho_HI[:,3] * 10 ** 8, xerr = [rho_HI[:,1] - rho_HI[:,0], rho_HI[:,2] - rho_HI[:,1]], yerr = 
                          [rho_HI[:,4] * 10 ** 8, rho_HI[:,5] * 10 ** 8], 
                          label = r'$\rho_{\text{HI}}$', fmt = 'ro', alpha = 0.7, 
                          elinewidth = 0.5, markersize = 4)
            self.axes.legend(prop = {'size': 14}, frameon = False)
            
        if use_walter:
            M_H2_W = lf.powerlaw(z_array, A = 1e7, B = 3, C = 2.3, D = 5.1)
                
            self.axes.plot(z_array, M_H2_W, lw = 2, color = 'purple',
                               label = r'Walter+2020')
            self.axes.legend(prop = {'size': 14}, frameon = False)
        
        self.axes.set_xlabel(r'Redshift', fontsize = 20)
        self.axes.set_ylabel(r'$\rho_{\text{H2}}$ [M$_\odot$ Mpc$^{-3}$]', fontsize = 20)
        self.axes.set_yscale('log')
        
        self.axes.grid()
        self.axes.tick_params(labelsize = 16, direction = 'in')
        
        
        self.draw()
        

class GasFraction(MplCanvas):
    def make_figure(self, alpha_CO1, alpha_CO2, alpha_HCN, a_CO, b_CO,
                    a_HCN, b_HCN, use_data = False):
        
        alpha_CO1 = float(alpha_CO1)
        alpha_CO2 = float(alpha_CO2)
        alpha_HCN = float(alpha_HCN)
        a_CO = float(a_CO)
        b_CO = float(b_CO)
        a_HCN = float(a_HCN)
        b_HCN = float(b_HCN)
        z_array = np.linspace(0, 10, 1000)
        
        self.axes.cla()
        
        M_dense = lf.HCN_density(z_array, a = a_HCN, b = b_HCN) * alpha_HCN
        
        M_H2 = lf.H2_density(z_array, a = a_CO, b = b_CO, alpha1 = alpha_CO1,
                             alpha2 = alpha_CO2)
        
        frac = M_dense / M_H2

        self.axes.plot(z_array, frac, '-k', lw = 2, 
                       label = r'Modelled gas fraction')
        
        if use_data:
            data_fn = '../Data/CO10-HCN10_cleaned.csv'
            data = pd.read_table(os.path.join(data_dir, data_fn),
                                 delimiter = ';')
            data = data.to_numpy(dtype = np.float64)
            data[np.isnan(data)] = 0
            
            Rybak_data_fn = 'Rybak22_data.csv'
            Rybak_data = np.loadtxt(os.path.join(data_dir, Rybak_data_fn),
                                    skiprows = 1, 
                                    usecols = (1, 2, 3, 4, 5))
            
            # redshifts
            z = data[0:21:2,0]


            L_CO = data[0:21:2,6]
            L_CO_err = data[0:21:2,7]

            L_HCN = data[1:22:2,6]
            L_HCN_err = data[1:22:2,7]

            M_dense_d = alpha_HCN * L_HCN

            idx = L_HCN_err != 99

            M_dense_d_err = L_HCN_err.copy()
            M_dense_d_err[idx] *= 10

            L_cutoff = lf.LIRtoLHCN(10**11, a = 1.12, b = 0.5)


            M_H2_d = L_CO.copy()
            M_H2_d_err = L_CO_err.copy()
            M_H2_d[L_CO <= L_cutoff ] *= alpha_CO1
            M_H2_d[L_CO > L_cutoff ] *= alpha_CO2
               
            M_H2_d_err[L_CO <= L_cutoff ] *= alpha_CO1
            M_H2_d_err[L_CO > L_cutoff ] *= alpha_CO2
               
            frac_d = M_dense_d / M_H2_d
            frac_d_err = np.zeros(len(frac_d))
            frac_d_err[idx] = np.sqrt(((1 / M_H2_d[idx]) * M_dense_d_err[idx]) ** 2 \
                            + (M_dense_d[idx] / (M_H2_d[idx] ** 2) * \
                                    M_H2_d_err[idx]) ** 2)
            frac_d[~idx] = M_dense_d[~idx] / (M_H2_d[~idx])


            Rybak_data = np.loadtxt('../Data/Rybak22_data.csv', skiprows = 1,
                                    usecols = (1,2,3,4,5))

            z_R = Rybak_data[:,0]
            L_CO_R = Rybak_data[:,1]
            L_CO_R_err = Rybak_data[:,2]
            L_HCN_R = Rybak_data[:,3]
            L_HCN_R_err = Rybak_data[:,4]


            M_dense_R = alpha_HCN * L_HCN_R

            idx_R = L_HCN_R_err != 99
            M_dense_R_err = L_HCN_R_err.copy()
            M_dense_R_err[idx_R] *= 10

            L_cutoff = lf.LIRtoLHCN(10**11, a = 1.12, b = 0.5)


            M_H2_R = L_CO_R.copy()
            M_H2_R_err = L_CO_R_err.copy()
            M_H2_R[L_CO_R <= L_cutoff ] *= alpha_CO1
            M_H2_R[L_CO_R > L_cutoff ] *= alpha_CO2

            M_H2_R_err[L_CO_R <= L_cutoff ] *= alpha_CO1
            M_H2_R_err[L_CO_R > L_cutoff ] *= alpha_CO2

            frac_R = M_dense_R / M_H2_R
            frac_R_err = np.sqrt(((1 / M_H2_R[idx_R]) * M_dense_R_err[idx_R]) ** 2 \
                                 + (M_dense_R[idx_R] / (M_H2_R[idx_R] ** 2) * \
                                    M_H2_R_err[idx_R]) ** 2)
            frac_R[~idx_R] = M_dense_R[~idx_R] / (M_H2_R[~idx_R])
            
            self.axes.errorbar(z_R[idx_R], frac_R[idx_R], yerr = frac_R_err,
                              label = r'DSFG', fmt = 'bo', alpha = 0.7,
                              elinewidth = 1, markersize = 5)
            self.axes.errorbar(z_R[~idx_R], frac_R[~idx_R], yerr = 0.1, uplims = True,
                              fmt = 'bo', alpha = 0.7,
                              elinewidth = 1, markersize = 5)

            QSO_idx = np.zeros(len(z), dtype = bool)
            QSO_idx[0:5] = True

            self.axes.errorbar(z[(idx) & (QSO_idx)], frac_d[(idx) & (QSO_idx)],
                          yerr = frac_d_err[(idx) & (QSO_idx)],
                          capsize = 0.3, label = r'QSO', fmt = 'g*', alpha = 0.7,
                          elinewidth = 1, markersize = 7)
            self.axes.errorbar(z[(~idx) & (QSO_idx)], frac_d[(~idx) & (QSO_idx)],
                          yerr = 0.1, uplims = True,
                          fmt = 'g*', alpha = 0.7,
                          elinewidth = 1, markersize = 7)

            self.axes.errorbar(z[(idx) & (~QSO_idx)], frac_d[(idx) & (~QSO_idx)],
                          yerr = frac_d_err[(idx) & (~QSO_idx)],
                          capsize = 0.3, label = r'SMG', fmt = 'r+', alpha = 0.7,
                          elinewidth = 1, markersize = 7)

            self.axes.errorbar(z[(~idx) & (~QSO_idx)], frac_d[(~idx) & (~QSO_idx)],
                          yerr = 0.1, uplims = True,
                          fmt = 'r+', alpha = 0.7,
                          elinewidth = 1, markersize = 7)
        
            self.axes.legend(prop = {'size': 14}, frameon = False)
            
            # self.axes.set_ylim(0, 1)
        
        self.axes.set_xlabel(r'Redshift', fontsize = 20)
        # self.axes.set_ylabel(r"L'$_{\text{HCN(1-0)}}$/L'$_{\text{CO(1-0)}}$",
        #                      fontsize = 20)
        self.axes.set_ylabel(r'Dense gas fraction [$\rho_{\text{dense}}$'+\
                             r'/$\rho_{\text{H2}}$]', fontsize = 20)
        #self.axes.set_yscale('log')
        
        
        self.axes.tick_params(labelsize = 16, direction = 'in')
        self.axes.grid()
        
        self.draw()
        
        
class LumFraction(MplCanvas):
    def make_figure(self, a_CO, b_CO, a_HCN, b_HCN, use_data = False):
        
        a_CO = float(a_CO)
        b_CO = float(b_CO)
        a_HCN = float(a_HCN)
        b_HCN = float(b_HCN)
        z_array = np.linspace(0, 10, 1000)
        
        self.axes.cla()
        
        L_HCN = lf.HCN_density(z_array, a = a_HCN, b = b_HCN)
        L_CO = lf.CO_density(z_array, a = a_CO, b = b_CO)
        
        frac = L_HCN / L_CO
        
        self.axes.plot(z_array, frac, '-k', lw = 2,
                       label = r'Modelled luminosity fraction')
        
        if use_data:
            data_fn = '../Data/CO10-HCN10_cleaned.csv'
            data = pd.read_table(os.path.join(data_dir, data_fn),
                                 delimiter = ';')
            data = data.to_numpy(dtype = np.float64)
            data[np.isnan(data)] = 0
            
            Rybak_data_fn = 'Rybak22_data.csv'
            Rybak_data = np.loadtxt(os.path.join(data_dir, Rybak_data_fn),
                                    skiprows = 1, 
                                    usecols = (1, 2, 3, 4, 5))
            
            # redshifts
            z = data[0:21:2,0]


            L_CO = data[0:21:2,6]
            L_CO_err = data[0:21:2,7]

            L_HCN = data[1:22:2,6]
            L_HCN_err = data[1:22:2,7]

            M_dense_d = L_HCN

            idx = L_HCN_err != 99

            M_dense_d_err = L_HCN_err.copy()
            M_dense_d_err[idx] *= 10

            L_cutoff = lf.LIRtoLHCN(10**11, a = 1.12, b = 0.5)


            M_H2_d = L_CO.copy()
            M_H2_d_err = L_CO_err.copy()
               
            frac_d = M_dense_d / M_H2_d
            frac_d_err = np.zeros(len(frac_d))
            frac_d_err[idx] = np.sqrt(((1 / M_H2_d[idx]) * M_dense_d_err[idx]) ** 2 \
                            + (M_dense_d[idx] / (M_H2_d[idx] ** 2) * \
                                    M_H2_d_err[idx]) ** 2)
            frac_d[~idx] = M_dense_d[~idx] / (M_H2_d[~idx])


            Rybak_data = np.loadtxt('../Data/Rybak22_data.csv', skiprows = 1,
                                    usecols = (1,2,3,4,5))

            z_R = Rybak_data[:,0]
            L_CO_R = Rybak_data[:,1]
            L_CO_R_err = Rybak_data[:,2]
            L_HCN_R = Rybak_data[:,3]
            L_HCN_R_err = Rybak_data[:,4]


            M_dense_R = L_HCN_R

            idx_R = L_HCN_R_err != 99
            M_dense_R_err = L_HCN_R_err.copy()
            M_dense_R_err[idx_R] *= 10



            M_H2_R = L_CO_R.copy()
            M_H2_R_err = L_CO_R_err.copy()



            frac_R = M_dense_R / M_H2_R
            frac_R_err = np.sqrt(((1 / M_H2_R[idx_R]) * M_dense_R_err[idx_R]) ** 2 \
                                 + (M_dense_R[idx_R] / (M_H2_R[idx_R] ** 2) * \
                                    M_H2_R_err[idx_R]) ** 2)
            frac_R[~idx_R] = M_dense_R[~idx_R] / (M_H2_R[~idx_R])
            
            self.axes.errorbar(z_R[idx_R], frac_R[idx_R], yerr = frac_R_err,
                              label = r'DSFG', fmt = 'bo', alpha = 0.7,
                              elinewidth = 1, markersize = 5)
            self.axes.errorbar(z_R[~idx_R], frac_R[~idx_R], yerr = 0.1, uplims = True,
                              fmt = 'bo', alpha = 0.7,
                              elinewidth = 1, markersize = 5)

            QSO_idx = np.zeros(len(z), dtype = bool)
            QSO_idx[0:5] = True

            self.axes.errorbar(z[(idx) & (QSO_idx)], frac_d[(idx) & (QSO_idx)],
                          yerr = frac_d_err[(idx) & (QSO_idx)],
                          capsize = 0.3, label = r'QSO', fmt = 'g*', alpha = 0.7,
                          elinewidth = 1, markersize = 7)
            self.axes.errorbar(z[(~idx) & (QSO_idx)], frac_d[(~idx) & (QSO_idx)],
                          yerr = 0.1, uplims = True,
                          fmt = 'g*', alpha = 0.7,
                          elinewidth = 1, markersize = 7)

            self.axes.errorbar(z[(idx) & (~QSO_idx)], frac_d[(idx) & (~QSO_idx)],
                          yerr = frac_d_err[(idx) & (~QSO_idx)],
                          capsize = 0.3, label = r'SMG', fmt = 'r+', alpha = 0.7,
                          elinewidth = 1, markersize = 7)

            self.axes.errorbar(z[(~idx) & (~QSO_idx)], frac_d[(~idx) & (~QSO_idx)],
                          yerr = 0.1, uplims = True,
                          fmt = 'r+', alpha = 0.7,
                          elinewidth = 1, markersize = 7)
        
            self.axes.legend(prop = {'size': 14}, frameon = False)
            
            self.axes.set_ylim(0, 0.6)
        
        
        self.axes.set_xlabel(r'Redshift', fontsize = 20)
        self.axes.set_ylabel(r"L'$_{\text{HCN(1-0)}}$/L'$_{\text{CO(1-0)}}$",
                              fontsize = 20)
        
        self.axes.tick_params(labelsize = 16, direction = 'in')
        self.axes.grid()
        
        self.draw()
        

class DepletionTime(MplCanvas):
    def make_figure(self, alpha_CO1, alpha_CO2, alpha_HCN, a_CO, b_CO,
                    a_HCN, b_HCN, use_data = False):
        
        alpha_CO1 = float(alpha_CO1)
        alpha_CO2 = float(alpha_CO2)
        alpha_HCN = float(alpha_HCN)
        a_CO = float(a_CO)
        b_CO = float(b_CO)
        a_HCN = float(a_HCN)
        b_HCN = float(b_HCN)
        z_array = np.linspace(0, 10, 1000)
        
        self.axes.cla()
        
        M_dense = lf.HCN_density(z_array, a = a_HCN, b = b_HCN) * alpha_HCN
        
        M_H2 = lf.H2_density(z_array, a = a_CO, b = b_CO, alpha1 = alpha_CO1,
                             alpha2 = alpha_CO2)
        
        tau_H2, tau_H2_err = lf.get_depletion_time(z_array, M_H2)
        tau_dense, tau_dense_err = lf.get_depletion_time(z_array, M_dense)
        
        if use_data:
            A = 1e7
            A_err = 0.14*1e5
            B = 3
            B_err = 0.6
            C = 2.3
            C_err = 0.3
            D = 5.1
            D_err = 0.5
            M_H2_W = lf.powerlaw(z_array, A = A, B = B,
                                 C = C, D = D)
                                 
            A_sd = lf.dpl_dA(z_array, A, B, C, D) * A_err
            B_sd = lf.dpl_dB(z_array, A, B, C, D) * B_err
            C_sd = lf.dpl_dC(z_array, A, B, C, D) * C_err
            D_sd = lf.dpl_dD(z_array, A, B, C, D) * D_err
            
            M_H2_W_err = np.sqrt(A_sd ** 2 + B_sd ** 2 + C_sd ** 2 + D_sd ** 2)
            M_H2_W_err = 0
            
            A_SFR = 0.0158
            A_SFR_err = 0.001
            B_SFR = 2.88
            B_SFR_err = 0.16
            C_SFR = 2.76
            C_SFR_err = 0.11
            D_SFR = 5.88
            D_SFR_err = 0.15
            
            cSFRd = lf.powerlaw(z_array, A = 0.0158, B = 2.88,
                                C = 2.75, D = 5.88)

            
            A_SFR_sd = lf.dpl_dA(z_array, A_SFR, B_SFR, C_SFR, D_SFR) * A_SFR_err
            B_SFR_sd = lf.dpl_dB(z_array, A_SFR, B_SFR, C_SFR, D_SFR) * B_SFR_err
            C_SFR_sd = lf.dpl_dC(z_array, A_SFR, B_SFR, C_SFR, D_SFR) * C_SFR_err
            D_SFR_sd = lf.dpl_dD(z_array, A_SFR, B_SFR, C_SFR, D_SFR) * D_SFR_err
            
            cSFRd_err = np.sqrt(A_SFR_sd ** 2 + B_SFR_sd ** 2 + C_SFR_sd ** 2 + D_SFR_sd ** 2)
            
            tau_W = M_H2_W / cSFRd
            
            tau_W_err = np.sqrt((1 / cSFRd * M_H2_W_err) ** 2 + \
                                (M_H2_W / (cSFRd ** 2) * cSFRd_err) ** 2)
                
            self.axes.plot(z_array, tau_W, '-g', lw = 2,
                           label = r'Walter+2020 $\tau_{\text{H2}}$')
            self.axes.fill_between(z_array, tau_W - tau_W_err,
                                   tau_W + tau_W_err, color = 'green',
                                   alpha = 0.5, lw = 0)
            
            
            
            
        
        self.axes.plot(z_array, tau_H2, '-k', lw = 2,
                       label = r'$\tau_{\text{H2}}$')
        self.axes.fill_between(z_array, tau_H2 - tau_H2_err, tau_H2 + tau_H2_err,
                               color = 'black', alpha = 0.5, lw = 0)
        
        self.axes.plot(z_array, tau_dense, '-b', lw = 2,
                       label = r'$\tau_{\text{dense}}$')
        self.axes.fill_between(z_array, tau_dense - tau_dense_err,
                               tau_dense + tau_dense_err, color = 'blue',
                               alpha = 0.5, lw = 0)
        
        self.axes.set_yscale('log')
        self.axes.set_xlabel(r'Redshift', fontsize = 20)
        self.axes.set_ylabel(r'$\tau_{\text{depl}}$ [yr]', fontsize = 20)
        
        self.axes.legend(prop = {'size': 14}, frameon = False)
        self.axes.grid()
        self.axes.tick_params(labelsize = 16, direction = 'in')
        
        self.draw()
        
        
        
