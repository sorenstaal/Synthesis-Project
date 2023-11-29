#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:05:47 2023

@author: sorenstaal
"""

import warnings
# to ignore scipy "numpy version" warning
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
    QSizePolicy,
    QMenuBar,
    QCheckBox,
    QHBoxLayout
)

os.chdir('/Users/sorenstaal/Documents/Uni/Kandidat/3. semester/Synthesis Project/GUI')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import utilities as utils

class FrontWindow(QWidget):

    def __init__(self):
        super().__init__()
            
        self.setWindowTitle("Synthesis Project")

        #top-level layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        
        # add text box
        self.label_1 = QLabel("Please select a plotting option:", self)
        self.label_1.setStyleSheet("padding: :15px")
        
        
        # create and connect combo box to swtich between pages
        self.pageCombo = QComboBox()
        item_options = ["H2 density", "Dense gas density",
                        "Dense gas fraction", "Luminosity fraction",
                        "Gas depletion time"]
        self.pageCombo.addItems(item_options)
        self.pageCombo.activated.connect(self.switchPage)
                      
        # create stacked layout for switching between plots/pages
        self.stackedLayout = QStackedLayout()
                      
        # Create first page - inputs for parameters. Combine two formlayouts in
        # a horizontal layout
        self.page1 = QWidget()
        self.page1Layout = QVBoxLayout(self.page1)
        self.page1HLayout = QHBoxLayout()
        self.page11Layout = QFormLayout()
        self.page12Layout = QFormLayout()
        
        self.alpha1 = QLineEdit("3.1")
        
        self.alpha2 = QLineEdit("0.8")
        self.a_val = QLineEdit("1.12")
        self.b_val = QLineEdit("0.5")
        self.a_sd = QLineEdit("0")
        self.b_sd = QLineEdit("0")
        
        self.page1.label = QLabel("L_IR-L_CO log-linear parameters")
        self.page1.empty_label = QLabel("")
        
        
        
        # check box for including data
        self.data1 = QCheckBox("Include H2 and HI data")
        self.walter1 = QCheckBox("Include Walter+2020 plot")
        
        # button for updating plot
        button1 = QPushButton("Plot current attributes", self)
        
        pl_rhoH2 = utils.MH2Plot(self.page1)
        
        pl_rhoH2.make_figure(self.alpha1.text(), self.alpha2.text(),
                             self.a_val.text(),self.b_val.text(),
                             self.a_sd.text(), self.b_sd.text())
        
        button1.clicked.connect(lambda: pl_rhoH2.make_figure(self.alpha1.text(), 
                                                             self.alpha2.text(),
                             self.a_val.text(), self.b_val.text(),
                             self.a_sd.text(), self.b_sd.text(),
                             self.data1.isChecked(),
                             self.walter1.isChecked()))
        
        self.page1.toolbar = NavigationToolbar(pl_rhoH2, self.page1)
        
        
        # page 1 layout
        self.page11Layout.addRow("α_CO_faint:", self.alpha1)
        self.page12Layout.addRow("α_CO_bright:", self.alpha2)
        
        self.page11Layout.addRow(self.page1.label)
        self.page12Layout.addRow(self.page1.empty_label)
        self.page11Layout.addRow("Parameter a:", self.a_val)
        self.page12Layout.addRow("Parameter b:", self.b_val)
        self.page11Layout.addRow("a scatter (%):", self.a_sd)
        self.page12Layout.addRow("b scatter (%):", self.b_sd)
        
        self.page1HLayout.addLayout(self.page11Layout)
        self.page1HLayout.addLayout(self.page12Layout)
        self.page1Layout.addLayout(self.page1HLayout)
        
        self.page1Layout.addWidget(self.data1)
        self.page1Layout.addWidget(self.walter1)
        self.page1Layout.addWidget(self.page1.toolbar)
        self.page1Layout.addWidget(pl_rhoH2)
        
        self.page1Layout.addWidget(button1)
        self.page1.setLayout(self.page1Layout)
        self.stackedLayout.addWidget(self.page1)
        
        
        #Create second page
        self.page2 = QWidget()
        self.page2Layout = QVBoxLayout(self.page2)
        self.page2HLayout = QHBoxLayout()
        self.page21Layout = QFormLayout()
        self.page22Layout = QFormLayout()
        alpha_HCN_inval = str(10)
        
        # Check box for including data
        self.data2 = QCheckBox("Include H2 and HI data")
        
        self.alpha_HCN = QLineEdit(alpha_HCN_inval)
        self.page2.a_HCN = QLineEdit("1")
        self.page2.b_HCN = QLineEdit("3")
        self.page2.a_sd = QLineEdit("0")
        self.page2.b_sd = QLineEdit("0")
        
        self.page2.label = QLabel("L_IR-L_HCN log-linear parameters")
        self.page2.empty_label1 = QLabel("")
        self.page2.empty_label2 = QLabel("")
        
        
        button2 = QPushButton("Plot current attributes", self)
        
        

        
        pl_rhoHCN = utils.MdensePlot(self.page2)
        
        pl_rhoHCN.make_figure(self.alpha_HCN.text(), self.page2.a_HCN.text(),
                              self.page2.b_HCN.text(), self.page2.a_sd.text(),
                              self.page2.b_sd.text())
        
        button2.clicked.connect(lambda: pl_rhoHCN.make_figure(self.alpha_HCN.text(),
                                                              self.page2.a_HCN.text(),
                                                              self.page2.b_HCN.text(),
                                                              self.page2.a_sd.text(),
                                                              self.page2.b_sd.text(),
                                                              self.data2.isChecked()))
        
        # add matplotlib toolbar
        self.page2.toolbar = NavigationToolbar(pl_rhoHCN, self.page2)
        
        
        # Page 2 layout
        self.page21Layout.addRow('α_HCN:',self.alpha_HCN)
        self.page21Layout.addRow(self.page2.label)
        self.page22Layout.addRow(self.page2.empty_label1)
        self.page22Layout.addRow(self.page2.empty_label2)
        self.page21Layout.addRow('Parameter a:',
                                 self.page2.a_HCN)
        self.page22Layout.addRow('Parameter b:',
                                 self.page2.b_HCN)
        self.page21Layout.addRow("a scatter (%):", self.page2.a_sd)
        self.page22Layout.addRow("b scatter (%):", self.page2.b_sd)
        
        self.page2HLayout.addLayout(self.page21Layout)
        self.page2HLayout.addLayout(self.page22Layout)
        self.page2Layout.addLayout(self.page2HLayout)
        self.page2Layout.addWidget(self.data2)
        self.page2Layout.addWidget(self.page2.toolbar)
        self.page2Layout.addWidget(pl_rhoHCN)
        self.page2Layout.addWidget(button2)
        self.page2.setLayout(self.page2Layout)
        self.stackedLayout.addWidget(self.page2)
        
        
        # Create page 3 (dense gas fraction)
        self.page3 = QWidget()
        self.page3Layout = QVBoxLayout(self.page3)
        self.page3HLayout = QHBoxLayout()
        self.page31Layout = QFormLayout()
        self.page32Layout = QFormLayout()
        
        self.page3.alpha_CO1 = QLineEdit("3.1")
        self.page3.alpha_CO2 = QLineEdit("0.8")
        self.page3.alpha_HCN = QLineEdit("10")
        
        self.page3.a_CO = QLineEdit("1.12")
        self.page3.b_CO = QLineEdit("0.5")
        self.page3.a_HCN = QLineEdit("1")
        self.page3.b_HCN = QLineEdit("3")
        
        self.page3.label1 = QLabel("L_IR-L_CO log-linear parameters")
        self.page3.label2 = QLabel("L_IR-L_HCN log-linear parameters")
        self.page3.empty_label1 = QLabel("")
        self.page3.empty_label2 = QLabel("")
        self.page3.empty_label3 = QLabel("")
        
        # check box for including data
        self.data3 = QCheckBox("Include gas fraction data")
        self.gaodata3 = QCheckBox("Include data from Gao+2004b")
        
        # button for updating plot
        button3 = QPushButton("Plot current attributes", self)
        
        # make dense gas fraction plot
        
        pl_gasfrac = utils.GasFraction(self.page3)
        
        pl_gasfrac.make_figure(self.page3.alpha_CO1.text(), self.page3.alpha_CO2.text(),
                            self.page3.alpha_HCN.text(), self.page3.a_CO.text(),
                            self.page3.b_CO.text(), self.page3.a_HCN.text(),
                            self.page3.b_HCN.text())
        
        # update figure
        button3.clicked.connect(lambda: pl_gasfrac.make_figure(self.page3.alpha_CO1.text(), 
                                                                 self.page3.alpha_CO2.text(),
                                                                 self.page3.alpha_HCN.text(), self.page3.a_CO.text(),
                                                                 self.page3.b_CO.text(), self.page3.a_HCN.text(),
                                                                 self.page3.b_HCN.text(),
                                                                 self.data3.isChecked(),
                                                                 self.gaodata3.isChecked()))
        # add toolbar
        self.page3.toolbar = NavigationToolbar(pl_gasfrac, self.page3)
        
        # Create page 3 layout
        self.page31Layout.addRow("α_CO_faint:", self.page3.alpha_CO1)
        self.page32Layout.addRow("α_CO_bright:", self.page3.alpha_CO2)
        self.page31Layout.addRow("α_HCN:", self.page3.alpha_HCN)
        self.page32Layout.addRow(self.page3.empty_label1)

        self.page31Layout.addRow(self.page3.label1)
        self.page32Layout.addRow(self.page3.empty_label2)        
        self.page31Layout.addRow("Parameter a:",
                                 self.page3.a_CO)
        self.page32Layout.addRow("Parameter b:",
                                 self.page3.b_CO)
        self.page31Layout.addRow(self.page3.label2)
        self.page32Layout.addRow(self.page3.empty_label3)
        self.page31Layout.addRow("Parameter a:",
                                 self.page3.a_HCN)
        self.page32Layout.addRow("Parameter b:",
                                 self.page3.b_HCN)
        
        self.page3HLayout.addLayout(self.page31Layout)
        self.page3HLayout.addLayout(self.page32Layout)
        self.page3Layout.addLayout(self.page3HLayout)
        
        self.page3Layout.addWidget(self.data3)
        self.page3Layout.addWidget(self.gaodata3)
        self.page3Layout.addWidget(self.page3.toolbar)
        self.page3Layout.addWidget(pl_gasfrac)
        
        self.page3Layout.addWidget(button3)
        self.page3.setLayout(self.page3Layout)
        self.stackedLayout.addWidget(self.page3)
        
        
        # Create page 4 (luminosity fraction)
        self.page4 = QWidget()
        self.page4Layout = QVBoxLayout(self.page4)
        self.page4HLayout = QHBoxLayout()
        self.page41Layout = QFormLayout()
        self.page42Layout = QFormLayout()
        
        self.page4.a_CO = QLineEdit("1.12")
        self.page4.b_CO = QLineEdit("0.5")
        self.page4.a_HCN = QLineEdit("1")
        self.page4.b_HCN = QLineEdit("3")
        
        self.page4.label1 = QLabel("L_IR-L_CO log-linear parameters")
        self.page4.label2 = QLabel("L_IR-L_HCN log-linear parameters")
        self.page4.empty_label1 = QLabel("")
        self.page4.empty_label2 = QLabel("")
        
        # check box for including data
        self.data4 = QCheckBox("Include luminosity fraction data")
        self.gaodata4 = QCheckBox("Include data from Gao+2004b")
        
        # plot for updating plot
        button4 = QPushButton("Plot current attributes", self)
        
        # make luminosity fraction plot
        pl_lumfrac = utils.LumFraction(self.page4)
        pl_lumfrac.make_figure(self.page4.a_CO.text(), self.page4.b_CO.text(),
                               self.page4.a_HCN.text(), self.page4.b_HCN.text())
        button4.clicked.connect(lambda: pl_lumfrac.make_figure(self.page4.a_CO.text(), 
                                                             self.page4.b_CO.text(),
                               self.page4.a_HCN.text(), self.page4.b_HCN.text(),
                               self.data4.isChecked(),
                               self.gaodata4.isChecked()))
        
        self.page4.toolbar = NavigationToolbar(pl_lumfrac, self.page4)
        
        # Create page 4 layout
        self.page41Layout.addRow(self.page4.label1)
        self.page42Layout.addRow(self.page4.empty_label1)
        self.page41Layout.addRow("Parameter a:", self.page4.a_CO)
        self.page42Layout.addRow("Parameter b:", self.page4.b_CO)
        self.page41Layout.addRow(self.page4.label2)
        self.page42Layout.addRow(self.page4.empty_label2)
        self.page41Layout.addRow("Parameter a:", self.page4.a_HCN)
        self.page42Layout.addRow("Parameter b:", self.page4.b_HCN)
        self.page4HLayout.addLayout(self.page41Layout)
        self.page4HLayout.addLayout(self.page42Layout)
        self.page4Layout.addLayout(self.page4HLayout)
        
        self.page4Layout.addWidget(self.data4)
        self.page4Layout.addWidget(self.gaodata4)
        self.page4Layout.addWidget(self.page4.toolbar)
        self.page4Layout.addWidget(pl_lumfrac)
        
        self.page4Layout.addWidget(button4)
        self.page4.setLayout(self.page4Layout)
        self.stackedLayout.addWidget(self.page4)
        
        
        
        # Page 5: gas depletion time
        self.page5 = QWidget()
        self.page5Layout = QVBoxLayout(self.page5)
        self.page5HLayout = QHBoxLayout()
        self.page51Layout = QFormLayout()
        self.page52Layout = QFormLayout()
        
        # input parameters
        self.page5.alpha_CO1 = QLineEdit("3.1")
        self.page5.alpha_CO2 = QLineEdit("0.8")
        self.page5.alpha_HCN = QLineEdit("10")
        
        self.page5.a_CO = QLineEdit("1.12")
        self.page5.b_CO = QLineEdit("0.5")
        self.page5.a_HCN = QLineEdit("1")
        self.page5.b_HCN = QLineEdit("3")
        
        self.page5.label1 = QLabel("L_IR-L_CO log-linear parameters")
        self.page5.label2 = QLabel("L_IR-L_HCN log-linear parameters")
        self.page5.empty_label1 = QLabel("")
        self.page5.empty_label2 = QLabel("")
        self.page5.empty_label3 = QLabel("")
        
        # Check box for including plot from Walters
        self.data5 = QCheckBox("Include Walter+2020")
        
        button5 = QPushButton("Plot current attributes", self)
        
        
        pl_depl = utils.DepletionTime(self.page5)
        
        pl_depl.make_figure(self.page5.alpha_CO1.text(), self.page5.alpha_CO2.text(),
                            self.page5.alpha_HCN.text(), self.page5.a_CO.text(),
                            self.page5.b_CO.text(), self.page5.a_HCN.text(),
                            self.page5.b_HCN.text())
        
        # update figure
        button5.clicked.connect(lambda: pl_depl.make_figure(self.page5.alpha_CO1.text(), 
                                                            self.page5.alpha_CO2.text(),
                            self.page5.alpha_HCN.text(), self.page5.a_CO.text(),
                            self.page5.b_CO.text(), self.page5.a_HCN.text(),
                            self.page5.b_HCN.text(),
                            self.data5.isChecked()))
        
        self.page5.toolbar = NavigationToolbar(pl_depl, self.page5)
        
        # create page 5 layout
        self.page51Layout.addRow("α_CO_faint:", self.page5.alpha_CO1)
        self.page52Layout.addRow("α_CO_bright:", self.page5.alpha_CO2)
        self.page51Layout.addRow("α_HCN:", self.page5.alpha_HCN)
        self.page52Layout.addRow(self.page5.empty_label1)
        
        self.page51Layout.addRow(self.page5.label1)
        self.page52Layout.addRow(self.page5.empty_label2)
        self.page51Layout.addRow("Parameter a:",
                                 self.page5.a_CO)
        self.page52Layout.addRow("Parameter b:",
                                 self.page5.b_CO)
        self.page51Layout.addRow(self.page5.label2)
        self.page52Layout.addRow(self.page5.empty_label2)
        self.page51Layout.addRow("Parameter a:",
                                 self.page5.a_HCN)
        self.page52Layout.addRow("Parameter b:",
                                 self.page5.b_HCN)
        
        self.page5HLayout.addLayout(self.page51Layout)
        self.page5HLayout.addLayout(self.page52Layout)
        self.page5Layout.addLayout(self.page5HLayout)
        
        self.page5Layout.addWidget(self.data5)
        self.page5Layout.addWidget(self.page5.toolbar)
        self.page5Layout.addWidget(pl_depl)
        
        self.page5Layout.addWidget(button5)
        self.page5.setLayout(self.page5Layout)
        self.stackedLayout.addWidget(self.page5)
        


        # Add combo box and stacked layout to top-level layout
        layout.addWidget(self.label_1)
        layout.addWidget(self.pageCombo)
        layout.addLayout(self.stackedLayout)
        
    
    # function for switching pages
    def switchPage(self):
        self.stackedLayout.setCurrentIndex(self.pageCombo.currentIndex())

